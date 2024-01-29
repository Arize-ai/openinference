import json
from abc import ABC
from inspect import signature
from typing import Any, Callable, Collection, Dict, Mapping, Tuple

from openinference.instrumentation.dspy.package import _instruments
from openinference.instrumentation.dspy.version import __version__
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

_DSPY_MODULE = "dspy"

# DSPy used to be called DSP - some of the modules still fall under the old namespace
_DSP_MODULE = "dsp"


class DSPyInstrumentor(BaseInstrumentor):  # type: ignore
    """
    OpenInference Instrumentor for DSPy
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)

        # Instrument LM (language model) calls
        from dsp.modules.lm import LM

        language_model_classes = LM.__subclasses__()
        for lm in language_model_classes:
            wrap_function_wrapper(
                module=_DSP_MODULE,
                name=lm.__name__ + ".basic_request",
                wrapper=_LMBasicRequestWrapper(tracer),
            )

        # Instrument DSPy constructs
        wrap_function_wrapper(
            module=_DSPY_MODULE,
            name="Predict.forward",
            wrapper=_PredictForwardWrapper(tracer),
        )

        wrap_function_wrapper(
            module=_DSPY_MODULE,
            name="Retrieve.__call__",
            wrapper=_RetrieverForwardWrapper(tracer),
        )

        wrap_function_wrapper(
            module=_DSPY_MODULE,
            name="Module.__call__",
            wrapper=_ModuleForwardWrapper(tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        from dsp.modules.lm import LM

        language_model_classes = LM.__subclasses__()
        for lm in language_model_classes:
            if hasattr(lm.request, "__wrapped__"):
                lm.request = lm.request.__wrapped__

        # Restore DSPy constructs
        from dspy import Predict

        if hasattr(Predict.forward, "__wrapped__"):
            Predict.forward = Predict.forward.__wrapped__


class _WithTracer(ABC):
    """
    Base class for wrappers that need a tracer. Acts as a trait for the wrappers
    """

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer


class _LMBasicRequestWrapper(_WithTracer):
    """
    Wrapper for DSP LM.basic_request
    Captures all calls to language models (lm)
    """

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        prompt = args[0]
        invocation_parameters = {**instance.kwargs, **kwargs}
        span_name = instance.__class__.__name__ + ".request"
        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                SpanAttributes.LLM_MODEL_NAME: instance.kwargs.get("model"),
                SpanAttributes.LLM_INVOCATION_PARAMETERS: json.dumps(invocation_parameters),
                SpanAttributes.INPUT_VALUE: str(prompt),
                SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT.value,
            },
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            # TODO: parse usage. Need to decide if this
            # instrumentation should be used in conjunction with model instrumentation
            span.set_attributes(
                {
                    SpanAttributes.OUTPUT_VALUE: json.dumps(response),
                    SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                }
            )
            span.set_status(trace_api.StatusCode.OK)
        return response


class _PredictForwardWrapper(_WithTracer):
    """
    A wrapper for the Predict class to have a chain span for each prediction
    """

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        signature = kwargs.get("signature", instance.signature)
        span_name = _get_predict_span_name(instance)
        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                SpanAttributes.INPUT_VALUE: json.dumps(
                    kwargs, cls=CustomJSONEncoder
                ),  # TODO: enhance custom encoder to handle templates
                SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
            },
        ) as span:
            try:
                prediction = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_attributes(
                {
                    SpanAttributes.OUTPUT_VALUE: json.dumps(
                        self._prediction_to_output_dict(prediction, signature)
                    ),
                    SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                }
            )
            span.set_status(trace_api.StatusCode.OK)
        return prediction

    def _prediction_to_output_dict(self, prediction: Any, signature: Any) -> Dict[str, Any]:
        """
        Parse the prediction fields to get the input and output fields
        """
        output = {}
        for field in signature.fields:
            if field.output_variable and field.output_variable in prediction:
                output[field.output_variable] = prediction.get(field.output_variable)
        return output


def _get_input_value(method: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """
    Parses a method call's inputs into a JSON string. Ensures a consistent
    output regardless of whether the those inputs are passed as positional or
    keyword arguments.
    """
    bound_arguments = signature(method).bind(
        None,  # the value bound to the method's self argument is discarded below, so pass None
        *args,
        **kwargs,
    )
    return json.dumps(
        {
            argument_name: argument_value
            for argument_name, argument_value in bound_arguments.arguments.items()
            if argument_name != "self"
        },
        cls=CustomJSONEncoder,
    )


class _ModuleForwardWrapper(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        span_name = instance.__class__.__name__ + ".forward"
        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                # At this time, dspy.Module does not have an abstract forward
                # method, but assumes that user-defined subclasses implement the
                # forward method.
                SpanAttributes.INPUT_VALUE: (
                    _get_input_value(forward_method, *args, **kwargs)
                    if (forward_method := getattr(instance.__class__, "forward", None))
                    else "{}"
                ),
                SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
            },
        ) as span:
            try:
                prediction = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_attributes(
                {
                    SpanAttributes.OUTPUT_VALUE: json.dumps(instance, cls=CustomJSONEncoder),
                    SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                }
            )
            span.set_status(trace_api.StatusCode.OK)
        return prediction


class _RetrieverForwardWrapper(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        span_name = instance.__class__.__name__ + ".forward"
        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
            },
        ) as span:
            try:
                prediction = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_attributes(
                {
                    SpanAttributes.OUTPUT_VALUE: json.dumps(instance, cls=CustomJSONEncoder),
                    SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                }
            )
            span.set_status(trace_api.StatusCode.OK)
        return prediction


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return repr(obj)


def _get_predict_span_name(instance: Any) -> str:
    class_name = instance.__class__.__name__
    if (signature := getattr(instance, "signature", None)) and (
        signature_name := getattr(signature, "__name__", None)
    ):
        return f"{class_name}.{signature_name}.forward"
    return class_name
