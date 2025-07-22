import json
from abc import ABC
from copy import copy, deepcopy
from enum import Enum
from inspect import signature
from logging import getLogger
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
)

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.trace import StatusCode
from opentelemetry.util.types import AttributeValue
from wrapt import BoundFunctionWrapper, FunctionWrapper, wrap_object

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    get_attributes_from_context,
    safe_json_dumps,
)
from openinference.instrumentation.dspy.package import _instruments
from openinference.instrumentation.dspy.version import __version__
from openinference.semconv.trace import (
    DocumentAttributes,
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

if TYPE_CHECKING:
    from dspy import LM

logger = getLogger(__name__)


_DSPY_MODULE = "dspy"


class DSPyInstrumentor(BaseInstrumentor):  # type: ignore
    """
    OpenInference Instrumentor for DSPy
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        self._tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        from dspy import Predict

        wrap_object(
            module="dspy",
            name="LM.__call__",
            factory=CopyableFunctionWrapper,
            args=(_LMCallWrapper(self._tracer),),
        )

        wrap_object(
            module="dspy",
            name="LM.acall",
            factory=CopyableFunctionWrapper,
            args=(_LMAcallWrapper(self._tracer),),
        )

        # Predict is a concrete (non-abstract) class that may be invoked
        # directly, but DSPy also has subclasses of Predict that override the
        # forward method. We instrument both the forward methods of the base
        # class and all subclasses.
        wrap_object(
            module=_DSPY_MODULE,
            name="Predict.forward",
            factory=CopyableFunctionWrapper,
            args=(_PredictForwardWrapper(self._tracer),),
        )
        wrap_object(
            module=_DSPY_MODULE,
            name="Predict.aforward",
            factory=CopyableFunctionWrapper,
            args=(_PredictAforwardWrapper(self._tracer),),
        )

        predict_subclasses = Predict.__subclasses__()
        for predict_subclass in predict_subclasses:
            wrap_object(
                module=_DSPY_MODULE,
                name=predict_subclass.__name__ + ".forward",
                factory=CopyableFunctionWrapper,
                args=(_PredictForwardWrapper(self._tracer),),
            )
            wrap_object(
                module=_DSPY_MODULE,
                name=predict_subclass.__name__ + ".aforward",
                factory=CopyableFunctionWrapper,
                args=(_PredictAforwardWrapper(self._tracer),),
            )

        wrap_object(
            module=_DSPY_MODULE,
            name="Retrieve.forward",
            factory=CopyableFunctionWrapper,
            args=(_RetrieverForwardWrapper(self._tracer),),
        )

        wrap_object(
            module=_DSPY_MODULE,
            # At this time, dspy.Module does not have an abstract forward
            # method, but assumes that user-defined subclasses implement the
            # forward method and invokes that method using __call__.
            name="Module.__call__",
            factory=CopyableFunctionWrapper,
            args=(_ModuleForwardWrapper(self._tracer),),
        )

        wrap_object(
            module=_DSPY_MODULE,
            # At this time, dspy.Module does not have an abstract forward
            # method, but assumes that user-defined subclasses implement the
            # forward method and invokes that method using __call__.
            name="Module.acall",
            factory=CopyableFunctionWrapper,
            args=(_ModuleAforwardWrapper(self._tracer),),
        )

        # At this time, there is no common parent class for retriever models as
        # there is for language models. We instrument the retriever models on a
        # case-by-case basis.
        wrap_object(
            module=_DSPY_MODULE,
            name="ColBERTv2.__call__",
            factory=CopyableFunctionWrapper,
            args=(_RetrieverModelCallWrapper(self._tracer),),
        )

        wrap_object(
            module=_DSPY_MODULE,
            name="Adapter.__call__",
            factory=CopyableFunctionWrapper,
            args=(_AdapterCallWrapper(self._tracer),),
        )

        wrap_object(
            module=_DSPY_MODULE,
            name="Adapter.acall",
            factory=CopyableFunctionWrapper,
            args=(_AdapterAcallWrapper(self._tracer),),
        )

        wrap_object(
            module=_DSPY_MODULE,
            name="Tool.__call__",
            factory=CopyableFunctionWrapper,
            args=(_ToolCallWrapper(self._tracer),),
        )

        wrap_object(
            module=_DSPY_MODULE,
            name="Tool.acall",
            factory=CopyableFunctionWrapper,
            args=(_AsyncToolCallWrapper(self._tracer),),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
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


class _LMCallWrapper(_WithTracer):
    """
    Wrapper for __call__ method on dspy.LM
    """

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: "LM",
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        arguments = _bind_arguments(wrapped, *args, **kwargs)
        span_name = instance.__class__.__name__ + ".__call__"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: LLM,
                        **dict(_input_value_and_mime_type(arguments)),
                        **dict(_llm_model_name(instance)),
                        **dict(_llm_provider(instance)),
                        **dict(_llm_invocation_parameters(instance, arguments)),
                        **dict(_llm_input_messages(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            response = wrapped(*args, **kwargs)
            span.set_status(StatusCode.OK)
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            **dict(_output_value_and_mime_type(response)),
                            **dict(_llm_output_messages(response)),
                        }
                    )
                )
            )
        return response


class _LMAcallWrapper(_WithTracer):
    """
    Wrapper for acall method on dspy.LM
    """

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: "LM",
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        arguments = _bind_arguments(wrapped, *args, **kwargs)
        span_name = instance.__class__.__name__ + ".acall"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: LLM,
                        **dict(_input_value_and_mime_type(arguments)),
                        **dict(_llm_model_name(instance)),
                        **dict(_llm_provider(instance)),
                        **dict(_llm_invocation_parameters(instance, arguments)),
                        **dict(_llm_input_messages(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            response = await wrapped(*args, **kwargs)
            span.set_status(StatusCode.OK)
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            **dict(_output_value_and_mime_type(response)),
                            **dict(_llm_output_messages(response)),
                        }
                    )
                )
            )
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
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        INPUT_MIME_TYPE: JSON,
                    }
                )
            ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            prediction = wrapped(*args, **kwargs)
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            OUTPUT_VALUE: safe_json_dumps(
                                self._prediction_to_output_dict(prediction, signature)
                            ),
                            OUTPUT_MIME_TYPE: JSON,
                        }
                    )
                )
            )
            span.set_status(StatusCode.OK)
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


class _PredictAforwardWrapper(_WithTracer):
    """
    A wrapper for the Predict class to have a chain span for each async prediction
    """

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
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
        has_overridden_aforward_method = getattr(cls, "aforward", None) is not getattr(
            Predict, "aforward", None
        )
        wrapped_method_is_base_class_aforward_method = (
            wrapped.__qualname__ == Predict.aforward.__qualname__
        )
        if (
            is_instance_of_predict_subclass
            and has_overridden_aforward_method
            and wrapped_method_is_base_class_aforward_method
        ):
            return await wrapped(*args, **kwargs)

        signature = kwargs.get("signature", instance.signature)
        span_name = _get_predict_span_name(instance)
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        INPUT_MIME_TYPE: JSON,
                    }
                )
            ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            prediction = await wrapped(*args, **kwargs)
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            OUTPUT_VALUE: safe_json_dumps(
                                self._prediction_to_output_dict(prediction, signature)
                            ),
                            OUTPUT_MIME_TYPE: JSON,
                        }
                    )
                )
            )
            span.set_status(StatusCode.OK)
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
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        # At this time, dspy.Module does not have an abstract forward
                        # method, but assumes that user-defined subclasses implement the
                        # forward method.
                        **(
                            {INPUT_VALUE: _get_input_value(forward_method, *args, **kwargs)}
                            if (forward_method := getattr(instance.__class__, "forward", None))
                            else {}
                        ),
                        INPUT_MIME_TYPE: JSON,
                    }
                )
            ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            prediction = wrapped(*args, **kwargs)
            span.set_attributes(
                dict(_flatten(_module_prediction_output_attributes(prediction, instance)))
            )
            span.set_status(StatusCode.OK)
        return prediction


class _ModuleAforwardWrapper(_WithTracer):
    """
    Instruments the acall method of dspy.Module. DSPy end users define custom
    subclasses of Module implementing a forward method, loosely resembling the
    ergonomics of torch.nn.Module. The acall method of dspy.Module invokes
    the forward method of the user-defined subclass.
    """

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        span_name = instance.__class__.__name__ + ".aforward"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        # At this time, dspy.Module does not have an abstract forward
                        # method, but assumes that user-defined subclasses implement the
                        # forward method.
                        **(
                            {INPUT_VALUE: _get_input_value(forward_method, *args, **kwargs)}
                            if (forward_method := getattr(instance.__class__, "aforward", None))
                            else {}
                        ),
                        INPUT_MIME_TYPE: JSON,
                    }
                )
            ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            prediction = await wrapped(*args, **kwargs)
            span.set_attributes(
                dict(_flatten(_module_prediction_output_attributes(prediction, instance)))
            )
            span.set_status(StatusCode.OK)
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
                        INPUT_MIME_TYPE: JSON,
                    }
                )
            ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            prediction = wrapped(*args, **kwargs)
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
            span.set_status(StatusCode.OK)
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
                        INPUT_MIME_TYPE: JSON,
                    }
                )
            ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            retrieved_documents = wrapped(*args, **kwargs)
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
            span.set_status(StatusCode.OK)
        return retrieved_documents


class _AdapterCallWrapper(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        arguments = _bind_arguments(wrapped, *args, **kwargs)
        span_name = instance.__class__.__name__ + ".__call__"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        **dict(_input_value_and_mime_type(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            response = wrapped(*args, **kwargs)
            span.set_status(StatusCode.OK)
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            **dict(_output_value_and_mime_type(response)),
                        }
                    )
                )
            )
        return response


class _AdapterAcallWrapper(_WithTracer):
    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        arguments = _bind_arguments(wrapped, *args, **kwargs)
        span_name = instance.__class__.__name__ + ".acall"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        **dict(_input_value_and_mime_type(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            response = await wrapped(*args, **kwargs)
            span.set_status(StatusCode.OK)
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            **dict(_output_value_and_mime_type(response)),
                        }
                    )
                )
            )
        return response


class _ToolCallWrapper(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        arguments = _bind_arguments(wrapped, *args, **kwargs)
        span_name = (instance.name or instance.__class__.__name__) + ".__call__"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: TOOL,
                        **dict(_input_value_and_mime_type(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            response = wrapped(*args, **kwargs)
            span.set_status(StatusCode.OK)
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            **dict(_output_value_and_mime_type(response)),
                        }
                    )
                )
            )
        return response


class _AsyncToolCallWrapper(_WithTracer):
    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        arguments = _bind_arguments(wrapped, *args, **kwargs)
        span_name = (instance.name or instance.__class__.__name__) + ".acall"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: TOOL,
                        **dict(_input_value_and_mime_type(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            response = await wrapped(*args, **kwargs)
            span.set_status(StatusCode.OK)
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            **dict(_output_value_and_mime_type(response)),
                        }
                    )
                )
            )
        return response


class DSPyJSONEncoder(json.JSONEncoder):
    """
    Provides support for non-JSON-serializable objects in DSPy.
    """

    def default(self, o: Any) -> Any:
        try:
            return super().default(o)
        except TypeError:
            from dspy.primitives.example import Example

            if hasattr(o, "_asdict"):
                # convert namedtuples to dictionaries
                return o._asdict()
            if isinstance(o, Example):
                # handles Prediction objects and other sub-classes of Example
                return getattr(o, "_store", {})
            return repr(o)


class SafeJSONEncoder(json.JSONEncoder):
    """
    Safely encodes non-JSON-serializable objects.
    """

    def default(self, o: Any) -> Any:
        try:
            return super().default(o)
        except TypeError:
            if hasattr(o, "dict") and callable(o.dict):  # pydantic v1 models, e.g., from Cohere
                return o.dict()
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
    return safe_json_dumps(
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


def _input_value_and_mime_type(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    yield INPUT_MIME_TYPE, JSON
    yield INPUT_VALUE, safe_json_dumps(arguments)


def _output_value_and_mime_type(response: Any) -> Iterator[Tuple[str, Any]]:
    yield OUTPUT_VALUE, safe_json_dumps(response)
    yield OUTPUT_MIME_TYPE, JSON


def parse_provider_and_model(model_str: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Parse a model string like 'openai/gpt-4', 'text-completion-openai/gpt-3.5-turbo-instruct/', etc.
    Returns (provider, model_name), both lowercased and stripped of trailing slashes.
    """
    if not isinstance(model_str, str) or not model_str:
        return None, None
    model_str = model_str.strip().rstrip("/")
    if "/" in model_str:
        provider, model_name = model_str.split("/", 1)
        provider = provider.strip().lower()
        model_name = model_name.strip()
        # Handle cases like 'text-completion-openai' -> 'openai'
        if "-" in provider:
            provider = provider.split("-")[-1]
        return provider, model_name
    return None, model_str.strip() if model_str else None


def _llm_model_name(lm: "LM") -> Iterator[Tuple[str, Any]]:
    if (model_name := getattr(lm, "model_name", None)) is not None:
        yield LLM_MODEL_NAME, model_name
        return
    elif (model := getattr(lm, "model", None)) is not None:
        provider, model_name = parse_provider_and_model(str(model))
        if model_name:
            yield LLM_MODEL_NAME, model_name


def _llm_provider(lm: "LM") -> Iterator[Tuple[str, Any]]:
    """
    Extract the LLM provider from a DSPy LM instance.

    This function attempts to extract the provider name through two strategies:
    1. From the provider attribute's class name (e.g., OpenAIProvider -> openai)
    2. Parsing from the model string (LiteLLM-style format like "openai/gpt-4")
    """
    # First try to get provider from the provider attribute
    if (provider := getattr(lm, "provider", None)) is not None:
        # Extract provider name from class name (e.g., OpenAIProvider -> openai)
        provider_class_name = provider.__class__.__name__.lower()
        if "provider" in provider_class_name and provider_class_name != "provider":
            provider_name = provider_class_name.removesuffix("provider")
            yield LLM_PROVIDER, provider_name
            return

    # Fallback: Parse provider from LiteLLM-style model string
    # LiteLLM uses format "provider/model" (e.g., "openai/gpt-4")
    # Also handles prefixed formats like "text-completion-openai/gpt-3.5-turbo-instruct"
    if (model := getattr(lm, "model", None)) is not None:
        provider, _ = parse_provider_and_model(str(model))
        if provider:
            yield LLM_PROVIDER, provider


def _llm_input_messages(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    if isinstance(prompt := arguments.get("prompt"), str):
        yield f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}", "user"
        yield f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}", prompt
    elif isinstance(messages := arguments.get("messages"), list):
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            if (role := message.get("role", None)) is not None:
                yield f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_ROLE}", role
            if (content := message.get("content", None)) is not None:
                yield f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_CONTENT}", content


def _llm_output_messages(response: Any) -> Iterator[Tuple[str, Any]]:
    if isinstance(response, Iterable):
        for i, message in enumerate(response):
            if isinstance(message, str):
                yield f"{LLM_OUTPUT_MESSAGES}.{i}.{MESSAGE_ROLE}", "assistant"
                yield f"{LLM_OUTPUT_MESSAGES}.{i}.{MESSAGE_CONTENT}", message


def _llm_invocation_parameters(lm: "LM", arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    lm_kwargs = _ if isinstance(_ := getattr(lm, "kwargs", {}), dict) else {}
    kwargs = _ if isinstance(_ := arguments.get("kwargs"), dict) else {}
    combined_kwargs = lm_kwargs | kwargs
    combined_kwargs.pop("api_key", None)
    yield LLM_INVOCATION_PARAMETERS, safe_json_dumps(combined_kwargs)


def _bind_arguments(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    method_signature = signature(method)
    bound_args = method_signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


def _module_prediction_output_attributes(prediction: Any, instance: Any) -> Dict[str, Any]:
    output_attributes = {OUTPUT_MIME_TYPE: JSON}
    import dspy

    if isinstance(prediction, dspy.Prediction):
        # https://github.com/stanfordnlp/dspy/blob/6fe693528323c9c10c82d90cb26711a985e18b29/dspy/primitives/example.py#L107C1-L108C1  # noqa E501
        # The Prediction object in DSPy works like a dictionary
        # https://github.com/stanfordnlp/dspy/blob/6fe693528323c9c10c82d90cb26711a985e18b29/dspy/primitives/prediction.py#L22  # noqa E501
        output_attributes[OUTPUT_VALUE] = safe_json_dumps(prediction.toDict())
    else:
        output_attributes[OUTPUT_VALUE] = safe_json_dumps(prediction, cls=DSPyJSONEncoder)
    return output_attributes


JSON = OpenInferenceMimeTypeValues.JSON.value
TEXT = OpenInferenceMimeTypeValues.TEXT.value
LLM = OpenInferenceSpanKindValues.LLM
TOOL = OpenInferenceSpanKindValues.TOOL.value
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
