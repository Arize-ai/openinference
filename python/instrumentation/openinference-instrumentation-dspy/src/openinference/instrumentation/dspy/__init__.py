import json
from abc import ABC
from copy import copy, deepcopy
from enum import Enum
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
)

import opentelemetry.context as context_api
from openinference.instrumentation.dspy.package import _instruments
from openinference.instrumentation.dspy.version import __version__
from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.util.types import AttributeValue
from typing_extensions import TypeGuard
from wrapt import BoundFunctionWrapper, FunctionWrapper, wrap_object

try:
    from google.generativeai.types import GenerateContentResponse  # type: ignore
except ImportError:
    GenerateContentResponse = None

if TYPE_CHECKING:
    from google.generativeai.types import (
        GenerateContentResponse as GenerateContentResponseType,
    )

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

        from dspy import Predict

        language_model_classes = LM.__subclasses__()
        for lm in language_model_classes:
            wrap_object(
                module=_DSP_MODULE,
                name=lm.__name__ + ".basic_request",
                factory=CopyableFunctionWrapper,
                args=(_LMBasicRequestWrapper(tracer),),
            )

        # Predict is a concrete (non-abstract) class that may be invoked
        # directly, but DSPy also has subclasses of Predict that override the
        # forward method. We instrument both the forward methods of the base
        # class and all subclasses.
        wrap_object(
            module=_DSPY_MODULE,
            name="Predict.forward",
            factory=CopyableFunctionWrapper,
            args=(_PredictForwardWrapper(tracer),),
        )

        predict_subclasses = Predict.__subclasses__()
        for predict_subclass in predict_subclasses:
            wrap_object(
                module=_DSPY_MODULE,
                name=predict_subclass.__name__ + ".forward",
                factory=CopyableFunctionWrapper,
                args=(_PredictForwardWrapper(tracer),),
            )

        wrap_object(
            module=_DSPY_MODULE,
            name="Retrieve.forward",
            factory=CopyableFunctionWrapper,
            args=(_RetrieverForwardWrapper(tracer),),
        )

        wrap_object(
            module=_DSPY_MODULE,
            # At this time, dspy.Module does not have an abstract forward
            # method, but assumes that user-defined subclasses implement the
            # forward method and invokes that method using __call__.
            name="Module.__call__",
            factory=CopyableFunctionWrapper,
            args=(_ModuleForwardWrapper(tracer),),
        )

        # At this time, there is no common parent class for retriever models as
        # there is for language models. We instrument the retriever models on a
        # case-by-case basis.
        wrap_object(
            module=_DSP_MODULE,
            name="ColBERTv2.__call__",
            factory=CopyableFunctionWrapper,
            args=(_RetrieverModelCallWrapper(tracer),),
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


class CopyableBoundFunctionWrapper(BoundFunctionWrapper):  # type: ignore
    """
    A bound function wrapper that can be copied and deep-copied. When used to
    wrap a class method, this allows the entire class to be copied and
    deep-copied.

    For reference, see
    https://github.com/GrahamDumpleton/wrapt/issues/86#issuecomment-426161271
    and
    https://wrapt.readthedocs.io/en/master/wrappers.html#custom-function-wrappers
    """

    def __copy__(self) -> "CopyableBoundFunctionWrapper":
        return CopyableBoundFunctionWrapper(
            copy(self.__wrapped__), self._self_instance, self._self_wrapper
        )

    def __deepcopy__(self, memo: Dict[Any, Any]) -> "CopyableBoundFunctionWrapper":
        return CopyableBoundFunctionWrapper(
            deepcopy(self.__wrapped__, memo), self._self_instance, self._self_wrapper
        )


class CopyableFunctionWrapper(FunctionWrapper):  # type: ignore
    """
    A function wrapper that can be copied and deep-copied. When used to wrap a
    class method, this allows the entire class to be copied and deep-copied.

    For reference, see
    https://github.com/GrahamDumpleton/wrapt/issues/86#issuecomment-426161271
    and
    https://wrapt.readthedocs.io/en/master/wrappers.html#custom-function-wrappers
    """

    __bound_function_wrapper__ = CopyableBoundFunctionWrapper

    def __copy__(self) -> "CopyableFunctionWrapper":
        return CopyableFunctionWrapper(copy(self.__wrapped__), self._self_wrapper)

    def __deepcopy__(self, memo: Dict[Any, Any]) -> "CopyableFunctionWrapper":
        return CopyableFunctionWrapper(deepcopy(self.__wrapped__, memo), self._self_wrapper)


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
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        prompt = args[0]
        invocation_parameters = {**instance.kwargs, **kwargs}
        span_name = instance.__class__.__name__ + ".request"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: LLM.value,
                        LLM_MODEL_NAME: instance.kwargs.get("model"),
                        LLM_INVOCATION_PARAMETERS: json.dumps(invocation_parameters),
                        INPUT_VALUE: str(prompt),
                        INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT.value,
                    }
                )
            ),
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
                dict(
                    _flatten(
                        {
                            OUTPUT_VALUE: _jsonify_output(response),
                            OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                        }
                    )
                )
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
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        from dspy import Predict

        # At this time, subclasses of Predict override the base class' forward
        # method and invoke the parent class' forward method from within the
        # overridden method. The forward method for both Predict and its
        # subclasses have been instrumented. To avoid creating duplicate spans
        # for a single invocation, we don't create a span for the base class'
        # forward method if the instance belongs to a proper subclass of Predict
        # with an overridden forward method.
        is_instance_of_predict_subclass = (
            isinstance(instance, Predict) and (cls := instance.__class__) is not Predict
        )
        has_overridden_forward_method = getattr(cls, "forward", None) is not getattr(
            Predict, "forward", None
        )
        wrapped_method_is_base_class_forward_method = (
            wrapped.__qualname__ == Predict.forward.__qualname__
        )
        if (
            is_instance_of_predict_subclass
            and has_overridden_forward_method
            and wrapped_method_is_base_class_forward_method
        ):
            return wrapped(*args, **kwargs)

        signature = kwargs.get("signature", instance.signature)
        span_name = _get_predict_span_name(instance)
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN.value,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                    }
                )
            ),
        ) as span:
            try:
                prediction = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            OUTPUT_VALUE: json.dumps(
                                self._prediction_to_output_dict(prediction, signature)
                            ),
                            OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                        }
                    )
                )
            )
            span.set_status(trace_api.StatusCode.OK)
        return prediction

    def _prediction_to_output_dict(self, prediction: Any, signature: Any) -> Dict[str, Any]:
        """
        Parse the prediction to get output fields
        """
        output = {}
        for output_field_name in signature.output_fields:
            if (prediction_value := prediction.get(output_field_name)) is not None:
                output[output_field_name] = prediction_value
        return output


class _ModuleForwardWrapper(_WithTracer):
    """
    Instruments the __call__ method of dspy.Module. DSPy end users define custom
    subclasses of Module implementing a forward method, loosely resembling the
    ergonomics of torch.nn.Module. The __call__ method of dspy.Module invokes
    the forward method of the user-defined subclass.
    """

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        span_name = instance.__class__.__name__ + ".forward"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN.value,
                        # At this time, dspy.Module does not have an abstract forward
                        # method, but assumes that user-defined subclasses implement the
                        # forward method.
                        **(
                            {INPUT_VALUE: _get_input_value(forward_method, *args, **kwargs)}
                            if (forward_method := getattr(instance.__class__, "forward", None))
                            else {}
                        ),
                        INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                    }
                )
            ),
        ) as span:
            try:
                prediction = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            OUTPUT_VALUE: json.dumps(prediction, cls=DSPyJSONEncoder),
                            OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                        }
                    )
                )
            )
            span.set_status(trace_api.StatusCode.OK)
        return prediction


class _RetrieverForwardWrapper(_WithTracer):
    """
    Instruments the forward method of dspy.Retrieve, which is a wrapper around
    retriever models such as ColBERTv2. At this time, Retrieve does not contain
    any additional information that cannot be gleaned from the underlying
    retriever model sub-span. It is, however, a user-facing concept, so we have
    decided to instrument it.
    """

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        span_name = instance.__class__.__name__ + ".forward"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: RETRIEVER.value,
                        INPUT_VALUE: _get_input_value(wrapped, *args, **kwargs),
                        INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                    }
                )
            ),
        ) as span:
            try:
                prediction = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            RETRIEVAL_DOCUMENTS: [
                                {
                                    DocumentAttributes.DOCUMENT_CONTENT: document_text,
                                }
                                for document_text in prediction.get("passages", [])
                            ],
                        }
                    )
                )
            )
            span.set_status(trace_api.StatusCode.OK)
        return prediction


class _RetrieverModelCallWrapper(_WithTracer):
    """
    Instruments the __call__ method of retriever models such as ColBERTv2.
    """

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        class_name = instance.__class__.__name__
        span_name = class_name + ".__call__"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: RETRIEVER.value,
                        INPUT_VALUE: (_get_input_value(wrapped, *args, **kwargs)),
                        INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                    }
                )
            ),
        ) as span:
            try:
                retrieved_documents = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            RETRIEVAL_DOCUMENTS: [
                                {
                                    DocumentAttributes.DOCUMENT_ID: doc["pid"],
                                    DocumentAttributes.DOCUMENT_CONTENT: doc["text"],
                                    DocumentAttributes.DOCUMENT_SCORE: doc["score"],
                                }
                                for doc in retrieved_documents
                            ],
                        }
                    )
                )
            )
            span.set_status(trace_api.StatusCode.OK)
        return retrieved_documents


class DSPyJSONEncoder(json.JSONEncoder):
    """
    Provides support for non-JSON-serializable objects in DSPy.
    """

    def default(self, o: Any) -> Any:
        try:
            return super().default(o)
        except TypeError:
            from dsp.templates.template_v3 import Template

            from dspy.primitives.example import Example

            if hasattr(o, "_asdict"):
                # convert namedtuples to dictionaries
                return o._asdict()
            if isinstance(o, Example):
                # handles Prediction objects and other sub-classes of Example
                return getattr(o, "_store", {})
            if isinstance(o, Template):
                return {
                    "fields": [self.default(field) for field in o.fields],
                    "instructions": o.instructions,
                }
            return repr(o)


def _get_input_value(method: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """
    Parses a method call's inputs into a JSON string. Ensures a consistent
    output regardless of whether the those inputs are passed as positional or
    keyword arguments.
    """

    # For typical class methods, the corresponding instance of inspect.Signature
    # does not include the self parameter. However, the inspect.Signature
    # instance for __call__ does include the self parameter.
    method_signature = signature(method)
    first_parameter_name = next(iter(method_signature.parameters), None)
    signature_contains_self_parameter = first_parameter_name in ["self"]
    bound_arguments = method_signature.bind(
        *(
            [None]  # the value bound to the method's self argument is discarded below, so pass None
            if signature_contains_self_parameter
            else []  # no self parameter, so no need to pass a value
        ),
        *args,
        **kwargs,
    )
    return json.dumps(
        {
            **{
                argument_name: argument_value
                for argument_name, argument_value in bound_arguments.arguments.items()
                if argument_name not in ["self", "kwargs"]
            },
            **bound_arguments.arguments.get("kwargs", {}),
        },
        cls=DSPyJSONEncoder,
    )


def _get_predict_span_name(instance: Any) -> str:
    """
    Gets the name for the Predict span, which are the composition of a Predict
    class or subclass and a user-defined signature. An example name would be
    "Predict(UserDefinedSignature).forward".
    """
    class_name = str(instance.__class__.__name__)
    if (signature := getattr(instance, "signature", None)) and (
        signature_name := _get_signature_name(signature)
    ):
        return f"{class_name}({signature_name}).forward"
    return f"{class_name}.forward"


def _get_signature_name(signature: Any) -> Optional[str]:
    """
    A best-effort attempt to get the name of a signature.
    """
    if (
        # At the time of this writing, the __name__ attribute on signatures does
        # not return the user-defined class name, but __qualname__ does.
        qual_name := getattr(signature, "__qualname__", None)
    ) is None:
        return None
    return str(qual_name.split(".")[-1])


def _flatten(mapping: Mapping[str, Any]) -> Iterator[Tuple[str, AttributeValue]]:
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in _flatten(value):
                yield f"{key}.{sub_key}", sub_value
        elif isinstance(value, List) and any(isinstance(item, Mapping) for item in value):
            for index, sub_mapping in enumerate(value):
                for sub_key, sub_value in _flatten(sub_mapping):
                    yield f"{key}.{index}.{sub_key}", sub_value
        else:
            if isinstance(value, Enum):
                value = value.value
            yield key, value


def _jsonify_output(response: Any) -> str:
    """
    Converts output to JSON string.
    """
    if _is_google_response(response):
        return json.dumps(_parse_google_response(response))
    return json.dumps(response)


def _is_google_response(response: Any) -> TypeGuard["GenerateContentResponseType"]:
    """
    Checks whether a candidate response is an instance of
    GenerateContentResponse returned by the Google generative AI SDK.
    """

    return GenerateContentResponse is not None and isinstance(response, GenerateContentResponse)


def _parse_google_response(response: "GenerateContentResponseType") -> Dict[str, Any]:
    """
    Parses a response from the Google generative AI SDK into a dictionary.
    """

    return {
        "text": response.text,
    }


OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER
CHAIN = OpenInferenceSpanKindValues.CHAIN
LLM = OpenInferenceSpanKindValues.LLM
INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
