import json
from abc import ABC
from copy import copy, deepcopy
from dataclasses import asdict, is_dataclass
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
from wrapt import BoundFunctionWrapper, FunctionWrapper, apply_patch, resolve_path, wrap_object

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
    ToolAttributes,
    ToolCallAttributes,
)

if TYPE_CHECKING:
    from dspy import LM

logger = getLogger(__name__)


_DSPY_MODULE = "dspy"


class DSPyInstrumentor(BaseInstrumentor):  # type: ignore
    """
    OpenInference Instrumentor for DSPy
    """

    def _wrap_object(
        self,
        module: str,
        name: str,
        factory: Callable[..., Any],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        wrap_object(
            module,
            name,
            factory,
            args,
            kwargs or {},
        )
        self._originals.append((module, name))

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        self._originals: list[tuple[str, str]] = []
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

        self._wrap_object(
            module="dspy",
            name="LM.__call__",
            factory=CopyableFunctionWrapper,
            args=(_LMCallWrapper(self._tracer),),  # type: ignore[arg-type]
        )

        self._wrap_object(
            module="dspy",
            name="LM.acall",
            factory=CopyableFunctionWrapper,
            args=(_LMAcallWrapper(self._tracer),),  # type: ignore[arg-type]
        )

        # The DummyLM class is the only typical subclass of DSPy's LM class.
        # It is used for unit testing and does not typically get used in
        # production, however it is helpful to instrument it to understand
        # program flow within testing scenarios.
        self._wrap_object(
            module="dspy.utils",
            name="DummyLM.__call__",
            factory=CopyableFunctionWrapper,
            args=(_LMCallWrapper(self._tracer),),  # type: ignore[arg-type]
        )

        self._wrap_object(
            module="dspy.utils",
            name="DummyLM.acall",
            factory=CopyableFunctionWrapper,
            args=(_LMAcallWrapper(self._tracer),),  # type: ignore[arg-type]
        )

        # Predict is a concrete (non-abstract) class that may be invoked
        # directly, but DSPy also has subclasses of Predict that override the
        # forward method. We instrument both the forward methods of the base
        # class and all subclasses.
        self._wrap_object(
            module=_DSPY_MODULE,
            name="Predict.forward",
            factory=CopyableFunctionWrapper,
            args=(_PredictForwardWrapper(self._tracer),),  # type: ignore[arg-type]
        )
        self._wrap_object(
            module=_DSPY_MODULE,
            name="Predict.aforward",
            factory=CopyableFunctionWrapper,
            args=(_PredictAforwardWrapper(self._tracer),),  # type: ignore[arg-type]
        )

        predict_subclasses = Predict.__subclasses__()
        for predict_subclass in predict_subclasses:
            self._wrap_object(
                module=_DSPY_MODULE,
                name=predict_subclass.__name__ + ".forward",
                factory=CopyableFunctionWrapper,
                args=(_PredictForwardWrapper(self._tracer),),  # type: ignore[arg-type]
            )
            self._wrap_object(
                module=_DSPY_MODULE,
                name=predict_subclass.__name__ + ".aforward",
                factory=CopyableFunctionWrapper,
                args=(_PredictAforwardWrapper(self._tracer),),  # type: ignore[arg-type]
            )

        self._wrap_object(
            module=_DSPY_MODULE,
            name="Retrieve.forward",
            factory=CopyableFunctionWrapper,
            args=(_RetrieverForwardWrapper(self._tracer),),  # type: ignore[arg-type]
        )

        self._wrap_object(
            module=_DSPY_MODULE,
            # At this time, dspy.Module does not have an abstract forward
            # method, but assumes that user-defined subclasses implement the
            # forward method and invokes that method using __call__.
            name="Module.__call__",
            factory=CopyableFunctionWrapper,
            args=(_ModuleForwardWrapper(self._tracer),),  # type: ignore[arg-type]
        )

        self._wrap_object(
            module=_DSPY_MODULE,
            # At this time, dspy.Module does not have an abstract forward
            # method, but assumes that user-defined subclasses implement the
            # forward method and invokes that method using __call__.
            name="Module.acall",
            factory=CopyableFunctionWrapper,
            args=(_ModuleAforwardWrapper(self._tracer),),  # type: ignore[arg-type]
        )

        # At this time, there is no common parent class for retriever models as
        # there is for language models. We instrument the retriever models on a
        # case-by-case basis.
        self._wrap_object(
            module=_DSPY_MODULE,
            name="ColBERTv2.__call__",
            factory=CopyableFunctionWrapper,
            args=(_RetrieverModelCallWrapper(self._tracer),),  # type: ignore[arg-type]
        )

        self._wrap_object(
            module=_DSPY_MODULE,
            name="Adapter.__call__",
            factory=CopyableFunctionWrapper,
            args=(_AdapterCallWrapper(self._tracer),),  # type: ignore[arg-type]
        )

        self._wrap_object(
            module=_DSPY_MODULE,
            name="Adapter.acall",
            factory=CopyableFunctionWrapper,
            args=(_AdapterAcallWrapper(self._tracer),),  # type: ignore[arg-type]
        )

        self._wrap_object(
            module=_DSPY_MODULE,
            name="Tool.__call__",
            factory=CopyableFunctionWrapper,
            args=(_ToolCallWrapper(self._tracer),),  # type: ignore[arg-type]
        )

        self._wrap_object(
            module=_DSPY_MODULE,
            name="Tool.acall",
            factory=CopyableFunctionWrapper,
            args=(_AsyncToolCallWrapper(self._tracer),),  # type: ignore[arg-type]
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        # Handle case where _uninstrument is called on fresh instance
        if not hasattr(self, "_originals"):
            return
        for module, name in self._originals[::-1]:
            unwrap_object(module, name)
        self._originals.clear()


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
        return CopyableBoundFunctionWrapper(  # type: ignore[call-arg]
            copy(self.__wrapped__), self._self_instance, self._self_wrapper
        )

    def __deepcopy__(self, memo: Dict[Any, Any]) -> "CopyableBoundFunctionWrapper":
        return CopyableBoundFunctionWrapper(  # type: ignore[call-arg]
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
        arguments = _normalize_lm_arguments(_bind_arguments(wrapped, *args, **kwargs))
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
                        **dict(_llm_tools(arguments)),
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
                            **dict(_llm_token_counts(response)),
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
        arguments = _normalize_lm_arguments(_bind_arguments(wrapped, *args, **kwargs))
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
                        **dict(_llm_tools(arguments)),
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
                            **dict(_llm_token_counts(response)),
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
    if _is_lm15_request(request := arguments.get("prompt")):
        arguments = {**arguments, "prompt": _lm15_request_value(request)}
    yield INPUT_MIME_TYPE, JSON
    yield INPUT_VALUE, safe_json_dumps(arguments)


def _output_value_and_mime_type(response: Any) -> Iterator[Tuple[str, Any]]:
    if _is_lm15_response(response):
        response = _lm15_response_value(response)
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
    if _is_lm15_request(prompt := arguments.get("prompt")):
        yield LLM_INPUT_MESSAGES, _lm15_request_messages(prompt)
    elif isinstance(prompt, str):
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
    if _is_lm15_response(response):
        yield LLM_OUTPUT_MESSAGES, [_lm15_message("assistant", response.message.parts)]
    elif isinstance(response, Iterable):
        for i, message in enumerate(response):
            if isinstance(message, str):
                yield f"{LLM_OUTPUT_MESSAGES}.{i}.{MESSAGE_ROLE}", "assistant"
                yield f"{LLM_OUTPUT_MESSAGES}.{i}.{MESSAGE_CONTENT}", message


def _llm_invocation_parameters(lm: "LM", arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    lm_kwargs = _ if isinstance(_ := getattr(lm, "kwargs", {}), dict) else {}
    kwargs = _ if isinstance(_ := arguments.get("kwargs"), dict) else {}
    if not kwargs and isinstance(_ := arguments.get("options"), dict):
        kwargs = _
    combined_kwargs = lm_kwargs | kwargs
    if _is_lm15_request(request := arguments.get("prompt")):
        combined_kwargs |= _lm15_config_value(getattr(request, "config", None))
    combined_kwargs.pop("api_key", None)
    yield LLM_INVOCATION_PARAMETERS, safe_json_dumps(combined_kwargs)


def _llm_tools(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    if not _is_lm15_request(request := arguments.get("prompt")):
        return
    for index, tool in enumerate(getattr(request, "tools", None) or ()):
        if getattr(tool, "type", None) != "function":
            continue
        schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        yield f"{LLM_TOOLS}.{index}.{TOOL_JSON_SCHEMA}", safe_json_dumps(schema)


def _llm_token_counts(response: Any) -> Iterator[Tuple[str, Any]]:
    if not _is_lm15_response(response):
        return
    usage = response.usage
    for attribute, field_name in (
        (LLM_TOKEN_COUNT_PROMPT, "input_tokens"),
        (LLM_TOKEN_COUNT_COMPLETION, "output_tokens"),
        (LLM_TOKEN_COUNT_TOTAL, "total_tokens"),
        (LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ, "cache_read_tokens"),
        (LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE, "cache_write_tokens"),
        (LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO, "input_audio_tokens"),
        (LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING, "reasoning_tokens"),
        (LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO, "output_audio_tokens"),
    ):
        if (count := getattr(usage, field_name, None)) is not None:
            yield attribute, count


# DSPy 3.5 replaced the OpenAI-style `messages`/`list[str]` LM interface with the
# provider-neutral `dspy.lm15.Request`/`Response` types. They are matched by shape
# rather than imported so that older DSPy versions keep working.
def _is_lm15_request(value: Any) -> bool:
    if isinstance(value, (str, Mapping)):
        return False
    return hasattr(value, "messages") and hasattr(value, "config")


def _is_lm15_response(value: Any) -> bool:
    return hasattr(value, "message") and hasattr(value, "usage") and hasattr(value, "finish_reason")


def _lm15_part_type(part: Any) -> Optional[str]:
    return getattr(part, "type", None)


def _lm15_text(parts: Any) -> Optional[str]:
    texts = [part.text for part in parts or () if _lm15_part_type(part) == "text"]
    return "".join(texts) if texts else None


def _lm15_message(role: str, parts: Any) -> Dict[str, Any]:
    tool_calls = [
        {
            TOOL_CALL_ID: part.id,
            TOOL_CALL_FUNCTION_NAME: part.name,
            TOOL_CALL_FUNCTION_ARGUMENTS_JSON: safe_json_dumps(part.input),
        }
        for part in parts or ()
        if _lm15_part_type(part) == "tool_call"
    ]
    return {
        MESSAGE_ROLE: role,
        MESSAGE_CONTENT: _lm15_text(parts),
        MESSAGE_TOOL_CALLS: tool_calls or None,
    }


def _lm15_request_messages(request: Any) -> List[Dict[str, Any]]:
    messages = []
    if system := getattr(request, "system", None):
        content = system if isinstance(system, str) else _lm15_text(system)
        messages.append({MESSAGE_ROLE: "system", MESSAGE_CONTENT: content})
    for message in request.messages:
        tool_results = [part for part in message.parts if _lm15_part_type(part) == "tool_result"]
        other_parts = [part for part in message.parts if _lm15_part_type(part) != "tool_result"]
        if other_parts or not tool_results:
            messages.append(_lm15_message(message.role, other_parts))
        for result in tool_results:
            messages.append(
                {
                    MESSAGE_ROLE: "tool",
                    MESSAGE_TOOL_CALL_ID: result.id,
                    MESSAGE_NAME: getattr(result, "name", None),
                    MESSAGE_CONTENT: _lm15_text(result.content),
                }
            )
    return messages


def _lm15_part_value(part: Any) -> Dict[str, Any]:
    part_type = _lm15_part_type(part)
    if part_type in ("text", "thinking", "refusal"):
        return {"type": part_type, "text": getattr(part, "text", None)}
    if part_type == "tool_call":
        return {"type": part_type, "id": part.id, "name": part.name, "input": part.input}
    if part_type == "tool_result":
        return {
            "type": part_type,
            "id": part.id,
            "name": getattr(part, "name", None),
            "content": [_lm15_part_value(content) for content in part.content],
        }
    # Media payloads can be large binary blobs; record only their kind.
    return {"type": part_type}


def _lm15_config_value(config: Any) -> Dict[str, Any]:
    if not is_dataclass(config) or isinstance(config, type):
        return {}
    return {key: value for key, value in asdict(config).items() if value not in (None, (), [], {})}


def _lm15_request_value(request: Any) -> Dict[str, Any]:
    system = getattr(request, "system", None)
    return {
        "model": getattr(request, "model", None),
        "system": system if system is None or isinstance(system, str) else _lm15_text(system),
        "messages": [
            {"role": message.role, "parts": [_lm15_part_value(part) for part in message.parts]}
            for message in request.messages
        ],
        "tools": [getattr(tool, "name", None) for tool in getattr(request, "tools", None) or ()],
        "config": _lm15_config_value(request.config),
    }


def _lm15_response_value(response: Any) -> Dict[str, Any]:
    usage = response.usage
    return {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "finish_reason": response.finish_reason,
        "message": {
            "role": response.message.role,
            "parts": [_lm15_part_value(part) for part in response.message.parts],
        },
        "usage": asdict(usage) if is_dataclass(usage) and not isinstance(usage, type) else None,
    }


def _bind_arguments(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    method_signature = signature(method)
    bound_args = method_signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


def _normalize_lm_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes bound `LM.__call__` / `LM.acall` arguments across DSPy versions.

    DSPy 3.3.0 changed the signature from `(prompt=None, messages=None, **kwargs)`
    to `(*items, prompt=None, messages=None, request=None, **kwargs)`, so a
    positionally-passed prompt string now arrives in `items` instead of `prompt`.
    Map that shape back to the legacy `prompt` argument so span attributes are
    stable across DSPy versions, and drop the new parameters when unused.
    """
    arguments = dict(arguments)
    items: Tuple[Any, ...] = tuple(arguments.pop("items", None) or ())
    if arguments.get("prompt") is None and len(items) == 1 and isinstance(items[0], str):
        arguments["prompt"] = items[0]
    elif items:
        arguments["items"] = list(items)
    if arguments.get("request", None) is None:
        arguments.pop("request", None)
    return arguments


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


def unwrap_object(module: Any, name: str) -> Optional[Any]:
    """
    Undo wrapt-style wrapping by restoring the original method
    that lives under the __wrapped__ chain.

    Args:
        module: class, instance, or module where the method was wrapped.
        name: attribute name to unwrap.

    Returns:
        The restored original attribute, or None if not wrapped.
    """
    parent, attribute_name, wrapped = resolve_path(module, name)

    if wrapped is None:
        logger.warning(f"{module}.{name} not found")
        return None

    # Start from the wrapped attribute
    original: Optional[Any] = getattr(wrapped, "__wrapped__", None)

    # Walk through chain: wrapper -> wrapper -> original
    while original is not None and hasattr(original, "__wrapped__"):
        original = getattr(original, "__wrapped__", None)
    if original is not None:
        apply_patch(parent, attribute_name, original)
    else:
        logger.warning(f"module: {module}, method: {name} not wrapped")
    return original


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
LLM_TOOLS = SpanAttributes.LLM_TOOLS
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE = (
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE
)
LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO
LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING = (
    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING
)
LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO
MESSAGE_NAME = MessageAttributes.MESSAGE_NAME
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
MESSAGE_TOOL_CALL_ID = MessageAttributes.MESSAGE_TOOL_CALL_ID
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA
