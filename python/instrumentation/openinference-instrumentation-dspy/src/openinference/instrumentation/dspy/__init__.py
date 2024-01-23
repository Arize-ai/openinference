import json
from abc import ABC
from typing import Any, Callable, Collection, Dict, Mapping, Tuple, Union

from openinference.instrumentation.dspy.package import _instruments
from openinference.instrumentation.dspy.version import __version__
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from wrapt import wrap_function_wrapper

_DSPY_MODULE = "dspy"

# DSPy used to be called DSP - some of the modules still fall under the old namespace
_DSP_MODULE = "dsp"


class DSPyInstrumentor(BaseInstrumentor):
    """
    OpenInference Instrumentor for DSPy
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
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

    def _uninstrument(self, **kwargs):
        from dsp.modules.lm import LM

        language_model_classes = LM.__subclasses__()
        for lm in language_model_classes:
            lm.request = lm.request.__wrapped__

        # Restore DSPy constructs
        from dspy import Predict

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
    ) -> None:
        print("LM.basic_request")
        prompt = args[0]
        kwargs = {**instance.kwargs, **kwargs}
        span_name = instance.__class__.__name__ + ".request"
        span = self._tracer.start_span(span_name)
        span.set_attributes(
            {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                SpanAttributes.LLM_MODEL_NAME: instance.kwargs.get("model"),
                SpanAttributes.LLM_INVOCATION_PARAMETERS: json.dumps(kwargs),
                SpanAttributes.INPUT_VALUE: prompt,
                SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT.value,
            }
        )
        response = wrapped(*args, **kwargs)
        # TODO: handle errors TODO: parse usage. Need to decide if this
        # instrumentation should be used in conjunction model instrumentation
        span.set_attributes(
            {
                SpanAttributes.OUTPUT_VALUE: json.dumps(response),
                SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
            }
        )
        span.set_status(trace_api.Status(trace_api.StatusCode.OK))
        span.end()
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
    ) -> None:
        print("Predict.forward")
        signature = kwargs.get("signature", instance.signature)
        span_name = signature.__class__.__name__ + ".forward"
        span = self._tracer.start_span(span_name)
        span.set_attributes(
            {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                SpanAttributes.INPUT_VALUE: json.dumps(kwargs),
                SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
            }
        )
        prediction = wrapped(*args, **kwargs)
        # TODO: handle errors
        span.set_attributes(
            {
                SpanAttributes.OUTPUT_VALUE: json.dumps(
                    self._prediction_to_output_dict(prediction, signature)
                ),
                SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
            }
        )
        span.set_status(trace_api.Status(trace_api.StatusCode.OK))
        span.end()
        return prediction

    def _prediction_to_output_dict(
        self, prediction, signature
    ) -> Dict[str, Union[str, int, float, bool]]:
        """
        Parse the prediction fields to get the input and output fields
        """
        output = {}
        for field in signature.fields:
            if field.output_variable and field.output_variable in prediction:
                output[field.output_variable] = prediction.get(field.output_variable)
        return output
