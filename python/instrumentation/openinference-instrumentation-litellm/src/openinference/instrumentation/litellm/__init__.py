import json
from functools import wraps
from typing import Any, Callable, Collection, Dict

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    get_attributes_from_context,
)
from openinference.instrumentation.litellm.version import __version__
from openinference.semconv.trace import (
    EmbeddingAttributes,
    ImageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.types import AttributeValue

import litellm


# Helper functions to set span attributes
def _set_span_attribute(span: trace_api.Span, name: str, value: AttributeValue) -> None:
    if value is not None and value != "":
        span.set_attribute(name, value)


def _instrument_func_type_completion(span: trace_api.Span, kwargs: Dict[str, Any]) -> None:
    """
    Currently instruments the functions:
        litellm.completion()
        litellm.acompletion() (async version of completion)
        litellm.completion_with_retries()
        litellm.acompletion_with_retries() (async version of completion_with_retries)
    """
    _set_span_attribute(
        span, SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value
    )
    _set_span_attribute(span, SpanAttributes.LLM_MODEL_NAME, kwargs.get("model", "unknown_model"))

    if "messages" in kwargs:
        _set_span_attribute(
            span, SpanAttributes.INPUT_VALUE, str(kwargs.get("messages")[0].get("content"))
        )
        for i, obj in enumerate(kwargs.get("messages")):
            for key, value in obj.items():
                _set_span_attribute(span, f"input.messages.{i}.{key}", value)

    invocation_params = {k: v for k, v in kwargs.items() if k not in ["model", "messages"]}
    _set_span_attribute(
        span, SpanAttributes.LLM_INVOCATION_PARAMETERS, json.dumps(invocation_params)
    )


def _instrument_func_type_embedding(span: trace_api.Span, kwargs: Dict[str, Any]) -> None:
    """
    Currently instruments the functions:
        litellm.embedding()
        litellm.aembedding() (async version of embedding)
    """
    _set_span_attribute(
        span,
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        OpenInferenceSpanKindValues.EMBEDDING.value,
    )
    _set_span_attribute(
        span, SpanAttributes.EMBEDDING_MODEL_NAME, kwargs.get("model", "unknown_model")
    )
    _set_span_attribute(span, EmbeddingAttributes.EMBEDDING_TEXT, kwargs.get("input"))
    _set_span_attribute(span, SpanAttributes.INPUT_VALUE, str(kwargs.get("input")))


def _instrument_func_type_image_generation(span: trace_api.Span, kwargs: Dict[str, Any]) -> None:
    """
    Currently instruments the functions:
        litellm.image_generation()
        litellm.aimage_generation() (async version of image_generation)
    """
    _set_span_attribute(
        span, SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value
    )
    _set_span_attribute(span, SpanAttributes.LLM_MODEL_NAME, kwargs.get("model"))
    _set_span_attribute(span, SpanAttributes.INPUT_VALUE, str(kwargs.get("prompt")))


def _finalize_span(span: trace_api.Span, result: Any) -> None:
    if isinstance(result, litellm.ModelResponse):
        _set_span_attribute(span, SpanAttributes.OUTPUT_VALUE, result.choices[0].message.content)
    elif isinstance(result, litellm.EmbeddingResponse):
        if len(result.data) > 0:
            first_embedding = result.data[0]
            _set_span_attribute(
                span,
                EmbeddingAttributes.EMBEDDING_VECTOR,
                json.dumps(first_embedding.get("embedding", [])),
            )
    elif isinstance(result, litellm.ImageResponse):
        if len(result.data) > 0:
            _set_span_attribute(span, ImageAttributes.IMAGE_URL, result.data[0]["url"])
            _set_span_attribute(span, SpanAttributes.OUTPUT_VALUE, result.data[0]["url"])
    if hasattr(result, "usage"):
        _set_span_attribute(
            span, SpanAttributes.LLM_TOKEN_COUNT_PROMPT, result.usage["prompt_tokens"]
        )
        _set_span_attribute(
            span, SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, result.usage["completion_tokens"]
        )
        _set_span_attribute(
            span, SpanAttributes.LLM_TOKEN_COUNT_TOTAL, result.usage["total_tokens"]
        )


class LiteLLMInstrumentor(BaseInstrumentor):
    original_litellm_funcs: Dict[
        str, Callable
    ] = {}  # Dictionary for original uninstrumented liteLLM functions

    # def __init__(self, tracer_provider: Optional[TracerProvider] = None, **kwargs):
    #     super().__init__()
    #     self.tracer_provider = tracer_provider
    #     if self.tracer_provider:
    #         trace.set_tracer_provider(self.tracer_provider)
    #     self.tracer = trace.get_tracer(__name__)
    #     # if not (config := kwargs.get("config")):
    #     #     config = TraceConfig()
    #     # else:
    #     #     assert isinstance(config, TraceConfig)
    #     # self.tracer = OITracer(
    #     #     trace.get_tracer(__name__, tracer_provider),
    #     #     config=config,
    #     # )

    __slots__ = ("tracer",)

    @wraps(litellm.completion)
    def _completion_wrapper(self, *args: Any, **kwargs: Any):
        with self.tracer.start_as_current_span(
            name="completion", attributes=dict(get_attributes_from_context())
        ) as span:
            _instrument_func_type_completion(span, kwargs)
            result = self.original_litellm_funcs["completion"](*args, **kwargs)
            _finalize_span(span, result)
        return result

    @wraps(litellm.acompletion)
    async def _acompletion_wrapper(self, *args: Any, **kwargs: Any):
        with self.tracer.start_as_current_span(
            name="acompletion", attributes=dict(get_attributes_from_context())
        ) as span:
            _instrument_func_type_completion(span, kwargs)
            result = await self.original_litellm_funcs["acompletion"](*args, **kwargs)
            _finalize_span(span, result)
        return result

    @wraps(litellm.completion_with_retries)
    def _completion_with_retries_wrapper(self, *args: Any, **kwargs: Any):
        with self.tracer.start_as_current_span(
            name="completion_with_retries", attributes=dict(get_attributes_from_context())
        ) as span:
            _instrument_func_type_completion(span, kwargs)
            result = self.original_litellm_funcs["completion_with_retries"](*args, **kwargs)
            _finalize_span(span, result)
        return result

    @wraps(litellm.acompletion_with_retries)
    async def _acompletion_with_retries_wrapper(self, *args: Any, **kwargs: Any):
        with self.tracer.start_as_current_span(
            name="acompletion_with_retries", attributes=dict(get_attributes_from_context())
        ) as span:
            _instrument_func_type_completion(span, kwargs)
            result = await self.original_litellm_funcs["acompletion_with_retries"](*args, **kwargs)
            _finalize_span(span, result)
        return result

    @wraps(litellm.embedding)
    def _embedding_wrapper(self, *args: Any, **kwargs: Any):
        with self.tracer.start_as_current_span(
            name="embedding", attributes=dict(get_attributes_from_context())
        ) as span:
            _instrument_func_type_embedding(span, kwargs)
            result = self.original_litellm_funcs["embedding"](*args, **kwargs)
            _finalize_span(span, result)
        return result

    @wraps(litellm.aembedding)
    async def _aembedding_wrapper(self, *args: Any, **kwargs: Any):
        with self.tracer.start_as_current_span(
            name="aembedding", attributes=dict(get_attributes_from_context())
        ) as span:
            _instrument_func_type_embedding(span, kwargs)
            result = await self.original_litellm_funcs["aembedding"](*args, **kwargs)
            _finalize_span(span, result)
        return result

    @wraps(litellm.image_generation)
    def _image_generation_wrapper(self, *args: Any, **kwargs: Any):
        with self.tracer.start_as_current_span(
            name="image_generation", attributes=dict(get_attributes_from_context())
        ) as span:
            _instrument_func_type_image_generation(span, kwargs)
            result = self.original_litellm_funcs["image_generation"](*args, **kwargs)
            _finalize_span(span, result)
        return result

    @wraps(litellm.aimage_generation)
    async def _aimage_generation_wrapper(self, *args: Any, **kwargs: Any):
        with self.tracer.start_as_current_span(
            name="aimage_generation", attributes=dict(get_attributes_from_context())
        ) as span:
            _instrument_func_type_image_generation(span, kwargs)
            result = await self.original_litellm_funcs["aimage_generation"](*args, **kwargs)
            _finalize_span(span, result)
        return result

    def _set_wrapper_attr(self, func_wrapper):
        func_wrapper.__func__.is_wrapper = True

    def _instrument(self, **kwargs: Any) -> None:
        print("INSTRUMENTING!!!")
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        self.tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        functions_to_instrument = {
            "completion": self._completion_wrapper,
            "acompletion": self._acompletion_wrapper,
            "completion_with_retries": self._completion_with_retries_wrapper,
            # Bug report filed on GitHub for acompletion_with_retries: https://github.com/BerriAI/litellm/issues/4908
            # "acompletion_with_retries": self._acompletion_with_retries_wrapper,
            "embedding": self._embedding_wrapper,
            "aembedding": self._aembedding_wrapper,
            "image_generation": self._image_generation_wrapper,
            "aimage_generation": self._aimage_generation_wrapper,
        }

        for func_name, func_wrapper in functions_to_instrument.items():
            if hasattr(litellm, func_name):
                original_func = getattr(litellm, func_name)
                self.original_litellm_funcs[func_name] = (
                    original_func  # Add original liteLLM function to dictionary
                )
                setattr(
                    litellm, func_name, func_wrapper
                )  # Monkey patch each function with their respective wrapper
                self._set_wrapper_attr(func_wrapper)

    def _uninstrument(self, **kwargs: Any) -> None:
        for func_name, original_func in LiteLLMInstrumentor.original_litellm_funcs.items():
            setattr(litellm, func_name, original_func)
        self.original_litellm_funcs.clear()

    def instrumentation_dependencies(self) -> Collection[str]:
        return ["litellm"]
