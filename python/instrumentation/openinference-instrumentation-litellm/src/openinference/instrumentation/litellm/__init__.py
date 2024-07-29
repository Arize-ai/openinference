import inspect
import json
from typing import Any, Callable, Collection, Dict, Optional

from openinference.semconv.trace import (
    EmbeddingAttributes,
    ImageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util.types import AttributeValue

import litellm


class LiteLLMInstrumentor(BaseInstrumentor):
    original_litellm_funcs: Dict[
        str, Callable
    ] = {}  # Dictionary for original uninstrumented liteLLM functions

    def __init__(self, tracer_provider: Optional[TracerProvider] = None):
        super().__init__()
        self.tracer_provider = tracer_provider
        if self.tracer_provider:
            trace.set_tracer_provider(self.tracer_provider)
        self.tracer = trace.get_tracer(__name__)

    def _set_span_attribute(self, span: trace.Span, name: str, value: AttributeValue) -> None:
        if value is not None and value != "":
            span.set_attribute(name, value)

    def _common_wrapper(self, func: Callable, span_name: str) -> Callable:
        def _handle_span(span: trace.Span, span_name: str, kwargs: Dict[str, Any]) -> None:
            if "embedding" in span_name:
                self._instrument_func_type_embedding(span, kwargs)
            elif "completion" in span_name:
                self._instrument_func_type_completion(span, kwargs)
            elif "image_generation" in span_name:
                self._instrument_func_type_image_generation(span, kwargs)

        def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with self.tracer.start_as_current_span(span_name) as span:
                _handle_span(span, span_name, kwargs)
                result = func(*args, **kwargs)
                self._finalize_span(span, result)
            return result

        async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with self.tracer.start_as_current_span(span_name) as span:
                _handle_span(span, span_name, kwargs)
                result = await func(*args, **kwargs)
                self._finalize_span(span, result)
            return result

        if inspect.iscoroutinefunction(func):
            return _async_wrapper
        else:
            return _sync_wrapper

    def _instrument_func_type_completion(self, span: trace.Span, kwargs: Dict[str, Any]) -> None:
        """
        Currently instruments the functions:
            litellm.completion()
            litellm.acompletion() (async version of completion)
            litellm.completion_with_retries()
            litellm.acompletion_with_retries() (async version of completion_with_retries)
        """
        self._set_span_attribute(
            span, SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value
        )
        self._set_span_attribute(
            span, SpanAttributes.LLM_MODEL_NAME, kwargs.get("model", "unknown_model")
        )

        if "messages" in kwargs:
            self._set_span_attribute(
                span, SpanAttributes.INPUT_VALUE, str(kwargs.get("messages")[0].get("content"))
            )
            for i, obj in enumerate(kwargs.get("messages")):
                for key, value in obj.items():
                    self._set_span_attribute(span, f"input.messages.{i}.{key}", value)

        invocation_params = {k: v for k, v in kwargs.items() if k not in ["model", "messages"]}
        self._set_span_attribute(
            span, SpanAttributes.LLM_INVOCATION_PARAMETERS, json.dumps(invocation_params)
        )

    def _instrument_func_type_embedding(self, span: trace.Span, kwargs: Dict[str, Any]) -> None:
        """
        Currently instruments the functions:
            litellm.embedding()
            litellm.aembedding() (async version of embedding)
        """
        self._set_span_attribute(
            span,
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.EMBEDDING.value,
        )
        self._set_span_attribute(
            span, SpanAttributes.EMBEDDING_MODEL_NAME, kwargs.get("model", "unknown_model")
        )
        self._set_span_attribute(span, EmbeddingAttributes.EMBEDDING_TEXT, kwargs.get("input"))
        self._set_span_attribute(span, SpanAttributes.INPUT_VALUE, str(kwargs.get("input")))

    def _instrument_func_type_image_generation(
        self, span: trace.Span, kwargs: Dict[str, Any]
    ) -> None:
        """
        Currently instruments the functions:
            litellm.image_generation()
            litellm.aimage_generation() (async version of image_generation)
        """
        self._set_span_attribute(
            span, SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value
        )
        self._set_span_attribute(span, SpanAttributes.LLM_MODEL_NAME, kwargs.get("model"))
        self._set_span_attribute(span, SpanAttributes.INPUT_VALUE, str(kwargs.get("prompt")))

    def _finalize_span(self, span: trace.Span, result: Any) -> None:
        if isinstance(result, litellm.ModelResponse):
            self._set_span_attribute(
                span, SpanAttributes.OUTPUT_VALUE, result.choices[0].message.content
            )
        elif isinstance(result, litellm.EmbeddingResponse):
            if len(result.data) > 0:
                first_embedding = result.data[0]
                self._set_span_attribute(
                    span,
                    EmbeddingAttributes.EMBEDDING_VECTOR,
                    json.dumps(first_embedding.get("embedding", [])),
                )
        elif isinstance(result, litellm.ImageResponse):
            if len(result.data) > 0:
                self._set_span_attribute(span, ImageAttributes.IMAGE_URL, result.data[0]["url"])
                self._set_span_attribute(span, SpanAttributes.OUTPUT_VALUE, result.data[0]["url"])
        if hasattr(result, "usage"):
            self._set_span_attribute(
                span, SpanAttributes.LLM_TOKEN_COUNT_PROMPT, result.usage["prompt_tokens"]
            )
            self._set_span_attribute(
                span, SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, result.usage["completion_tokens"]
            )
            self._set_span_attribute(
                span, SpanAttributes.LLM_TOKEN_COUNT_TOTAL, result.usage["total_tokens"]
            )

    def _instrument(self, tracer_provider: Optional[TracerProvider] = None) -> None:
        if tracer_provider:
            self.tracer_provider = tracer_provider
            trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(__name__)

        functions_to_instrument = {
            "completion": "completion",
            "acompletion": "acompletion",
            "completion_with_retries": "completion_with_retries",
            # Bug report filed on GitHub for acompletion_with_retries: https://github.com/BerriAI/litellm/issues/4908
            # 'acompletion_with_retries': 'acompletion_with_retries',
            "embedding": "embedding",
            "aembedding": "aembedding",
            "image_generation": "image_generation",
            "aimage_generation": "aimage_generation",
        }

        for func_name, span_name in functions_to_instrument.items():
            if hasattr(litellm, func_name):
                original_func = getattr(litellm, func_name)
                LiteLLMInstrumentor.original_litellm_funcs[func_name] = (
                    original_func  # Add original liteLLM function to dictionary
                )
                setattr(litellm, func_name, self._common_wrapper(original_func, span_name))

    def _uninstrument(self, **kwargs: Any) -> None:
        for func_name, original_func in LiteLLMInstrumentor.original_litellm_funcs.items():
            setattr(litellm, func_name, original_func)
        LiteLLMInstrumentor.original_litellm_funcs.clear()

    def instrumentation_dependencies(self) -> Collection[str]:
        return ["litellm"]

    def instrument(self, **kwargs: Any) -> None:
        super().instrument(**kwargs)

    def uninstrument(self, **kwargs: Any) -> None:
        super().uninstrument(**kwargs)
