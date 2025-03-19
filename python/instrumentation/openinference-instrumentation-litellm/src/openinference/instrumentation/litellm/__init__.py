import json
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Tuple,
    TypeVar,
    Union,
)

from openai.types.image import Image
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.util.types import AttributeValue

import litellm
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import (
    Choices,
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
)
from litellm.types.utils import (
    Message as LitellmMessage,
)
from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    get_attributes_from_context,
    safe_json_dumps,
)
from openinference.instrumentation.litellm.package import _instruments
from openinference.instrumentation.litellm.version import __version__
from openinference.semconv.trace import (
    EmbeddingAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


# Helper functions to set span attributes
def _set_span_attribute(span: trace_api.Span, name: str, value: AttributeValue) -> None:
    if value is not None and value != "":
        span.set_attribute(name, value)


T = TypeVar("T", bound=type)


def is_iterable_of(lst: Iterable[object], tp: T) -> bool:
    return isinstance(lst, Iterable) and all(isinstance(x, tp) for x in lst)


def _get_attributes_from_message_param(
    message: Union[Mapping[str, Any], LitellmMessage],
) -> Iterator[Tuple[str, AttributeValue]]:
    if not hasattr(message, "get"):
        return
    if role := message.get("role"):
        yield (
            MessageAttributes.MESSAGE_ROLE,
            role.value if isinstance(role, Enum) else role,
        )

    if content := message.get("content"):
        if isinstance(content, str):
            yield MessageAttributes.MESSAGE_CONTENT, content
        elif is_iterable_of(content, dict):
            for index, c in list(enumerate(content)):
                for key, value in _get_attributes_from_message_content(c):
                    yield f"{MessageAttributes.MESSAGE_CONTENTS}.{index}.{key}", value


def _get_attributes_from_message_content(
    content: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    content = dict(content)
    type_ = content.pop("type")
    if type_ == "text":
        yield f"{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text"
        if text := content.pop("text"):
            yield f"{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", text
    elif type_ == "image_url":
        yield f"{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "image"
        if image := content.pop("image_url"):
            for key, value in _get_attributes_from_image(image):
                yield f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{key}", value


def _get_attributes_from_image(
    image: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    image = dict(image)
    if url := image.pop("url"):
        yield f"{ImageAttributes.IMAGE_URL}", url


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

    if messages := kwargs.get("messages"):
        messages_as_dicts = []
        for input_message in messages:
            if isinstance(input_message, LitellmMessage):
                messages_as_dicts.append(input_message.json())  # type: ignore[no-untyped-call]
            else:
                messages_as_dicts.append(input_message)

        for index, input_message in enumerate(messages):
            for key, value in _get_attributes_from_message_param(input_message):
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{key}", value
                )

        if messages_as_dicts:
            _set_span_attribute(
                span, SpanAttributes.INPUT_VALUE, safe_json_dumps({"messages": messages_as_dicts})
            )
            _set_span_attribute(span, SpanAttributes.INPUT_MIME_TYPE, "application/json")

    invocation_params = {k: v for k, v in kwargs.items() if k not in ["model", "messages"]}
    _set_span_attribute(
        span, SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_params)
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
    _set_span_attribute(span, EmbeddingAttributes.EMBEDDING_TEXT, str(kwargs.get("input")))
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
    if model := kwargs.get("model"):
        _set_span_attribute(span, SpanAttributes.LLM_MODEL_NAME, model)
    if prompt := kwargs.get("prompt"):
        _set_span_attribute(span, SpanAttributes.INPUT_VALUE, str(prompt))


def _finalize_span(span: trace_api.Span, result: Any) -> None:
    if isinstance(result, ModelResponse):
        for idx, choice in enumerate(result.choices):
            if not isinstance(choice, Choices):
                continue

            if idx == 0 and choice.message and (output := choice.message.content):
                _set_span_attribute(span, SpanAttributes.OUTPUT_VALUE, output)

            for key, value in _get_attributes_from_message_param(choice.message):
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{idx}.{key}", value
                )

    elif isinstance(result, EmbeddingResponse):
        if result_data := result.data:
            first_embedding = result_data[0]
            _set_span_attribute(
                span,
                EmbeddingAttributes.EMBEDDING_VECTOR,
                json.dumps(first_embedding.get("embedding", [])),
            )
    elif isinstance(result, ImageResponse):
        if len(result.data) > 0:
            if img_data := result.data[0]:
                if isinstance(img_data, Image) and (url := (img_data.url or img_data.b64_json)):
                    _set_span_attribute(span, ImageAttributes.IMAGE_URL, url)
                    _set_span_attribute(span, SpanAttributes.OUTPUT_VALUE, url)
                elif isinstance(img_data, dict) and (
                    url := (img_data.get("url") or img_data.get("b64_json"))
                ):
                    _set_span_attribute(span, ImageAttributes.IMAGE_URL, url)
                    _set_span_attribute(span, SpanAttributes.OUTPUT_VALUE, url)
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


def _finalize_sync_streaming_span(span: trace_api.Span, stream: CustomStreamWrapper) -> Any:
    output_messages: Dict[int, Dict[str, Any]] = {}
    usage_stats = None
    try:
        for token in stream:
            if token.choices:
                for choice in token.choices:
                    idx = choice.index
                    if idx not in output_messages:
                        output_messages[idx] = {"role": None, "content": ""}
                    delta = choice.delta
                    if delta:
                        role = getattr(delta, "role", None)
                        content = getattr(delta, "content", None)
                        if role is not None and output_messages[idx]["role"] is None:
                            output_messages[idx]["role"] = role
                        if content is not None:
                            output_messages[idx]["content"] += content
            if getattr(token, "usage", None):
                usage_stats = token.usage
            yield token
        aggregated_output = output_messages.get(0, {}).get("content", "")
        _set_span_attribute(span, SpanAttributes.OUTPUT_VALUE, aggregated_output)
        for idx, msg in output_messages.items():
            message = {"role": msg.get("role"), "content": msg.get("content")}
            for key, value in _get_attributes_from_message_param(message):
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{idx}.{key}", value
                )

        if usage_stats:
            _set_span_attribute(
                span, SpanAttributes.LLM_TOKEN_COUNT_PROMPT, usage_stats.get("prompt_tokens")
            )
            _set_span_attribute(
                span,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                usage_stats.get("completion_tokens"),
            )
            _set_span_attribute(
                span, SpanAttributes.LLM_TOKEN_COUNT_TOTAL, usage_stats.get("total_tokens")
            )
    except Exception as e:
        span.record_exception(e)
        raise
    finally:
        span.end()


async def _finalize_streaming_span(span: trace_api.Span, stream: CustomStreamWrapper) -> Any:
    output_messages: Dict[int, Dict[str, Any]] = {}
    usage_stats = None
    try:
        async for token in stream:
            if token.choices:
                for choice in token.choices:
                    idx = choice.index
                    if idx not in output_messages:
                        output_messages[idx] = {"role": None, "content": ""}
                    delta = choice.delta
                    if delta:
                        role = getattr(delta, "role", None)
                        content = getattr(delta, "content", None)
                        if role is not None and output_messages[idx]["role"] is None:
                            output_messages[idx]["role"] = role
                        if content is not None:
                            output_messages[idx]["content"] += content
            if getattr(token, "usage", None):
                usage_stats = token.usage
            yield token
        aggregated_output = output_messages.get(0, {}).get("content", "")
        _set_span_attribute(span, SpanAttributes.OUTPUT_VALUE, aggregated_output)
        for idx, msg in output_messages.items():
            message = {"role": msg.get("role"), "content": msg.get("content")}
            for key, value in _get_attributes_from_message_param(message):
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{idx}.{key}", value
                )

        if usage_stats:
            _set_span_attribute(
                span, SpanAttributes.LLM_TOKEN_COUNT_PROMPT, usage_stats.get("prompt_tokens")
            )
            _set_span_attribute(
                span,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                usage_stats.get("completion_tokens"),
            )
            _set_span_attribute(
                span, SpanAttributes.LLM_TOKEN_COUNT_TOTAL, usage_stats.get("total_tokens")
            )
    except Exception as e:
        span.record_exception(e)
        raise
    finally:
        span.end()


class LiteLLMInstrumentor(BaseInstrumentor):  # type: ignore
    original_litellm_funcs: Dict[
        str, Callable[..., Any]
    ] = {}  # Dictionary for original uninstrumented liteLLM functions

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

    @wraps(litellm.completion)
    def _completion_wrapper(self, *args: Any, **kwargs: Any) -> ModelResponse:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return self.original_litellm_funcs["completion"](*args, **kwargs)  # type:ignore

        result = self.original_litellm_funcs["completion"](*args, **kwargs)

        if isinstance(result, CustomStreamWrapper) and kwargs.get("stream", False):
            span = self._tracer.start_span(
                name="completion", attributes=dict(get_attributes_from_context())
            )
            _instrument_func_type_completion(span, kwargs)
            return _finalize_sync_streaming_span(span, result)  # type:ignore
        else:
            with self._tracer.start_as_current_span(
                name="completion", attributes=dict(get_attributes_from_context())
            ) as span:
                _instrument_func_type_completion(span, kwargs)
                _finalize_span(span, result)
            return result  # type:ignore

    @wraps(litellm.acompletion)
    async def _acompletion_wrapper(
        self, *args: Any, **kwargs: Any
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await self.original_litellm_funcs["acompletion"](*args, **kwargs)  # type:ignore

        result = await self.original_litellm_funcs["acompletion"](*args, **kwargs)

        if hasattr(result, "__aiter__"):
            span = self._tracer.start_span(
                name="acompletion", attributes=dict(get_attributes_from_context())
            )
            _instrument_func_type_completion(span, kwargs)
            return _finalize_streaming_span(span, result)  # type:ignore
        else:
            with self._tracer.start_as_current_span(
                name="acompletion", attributes=dict(get_attributes_from_context())
            ) as span:
                _instrument_func_type_completion(span, kwargs)
                _finalize_span(span, result)
                return result  # type:ignore

    @wraps(litellm.completion_with_retries)
    def _completion_with_retries_wrapper(self, *args: Any, **kwargs: Any) -> ModelResponse:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return self.original_litellm_funcs["completion_with_retries"](*args, **kwargs)  # type:ignore
        with self._tracer.start_as_current_span(
            name="completion_with_retries", attributes=dict(get_attributes_from_context())
        ) as span:
            _instrument_func_type_completion(span, kwargs)
            result = self.original_litellm_funcs["completion_with_retries"](*args, **kwargs)
            _finalize_span(span, result)
        return result  # type:ignore

    @wraps(litellm.acompletion_with_retries)
    async def _acompletion_with_retries_wrapper(self, *args: Any, **kwargs: Any) -> ModelResponse:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return self.original_litellm_funcs["acompletion_with_retries"](*args, **kwargs)  # type:ignore
        with self._tracer.start_as_current_span(
            name="acompletion_with_retries", attributes=dict(get_attributes_from_context())
        ) as span:
            _instrument_func_type_completion(span, kwargs)
            result = await self.original_litellm_funcs["acompletion_with_retries"](*args, **kwargs)
            _finalize_span(span, result)
        return result  # type:ignore

    @wraps(litellm.embedding)
    def _embedding_wrapper(self, *args: Any, **kwargs: Any) -> EmbeddingResponse:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return self.original_litellm_funcs["embedding"](*args, **kwargs)  # type:ignore
        with self._tracer.start_as_current_span(
            name="embedding", attributes=dict(get_attributes_from_context())
        ) as span:
            _instrument_func_type_embedding(span, kwargs)
            result = self.original_litellm_funcs["embedding"](*args, **kwargs)
            _finalize_span(span, result)
        return result  # type:ignore

    @wraps(litellm.aembedding)
    async def _aembedding_wrapper(self, *args: Any, **kwargs: Any) -> EmbeddingResponse:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return self.original_litellm_funcs["aembedding"](*args, **kwargs)  # type:ignore
        with self._tracer.start_as_current_span(
            name="aembedding", attributes=dict(get_attributes_from_context())
        ) as span:
            _instrument_func_type_embedding(span, kwargs)
            result = await self.original_litellm_funcs["aembedding"](*args, **kwargs)
            _finalize_span(span, result)
        return result  # type:ignore

    @wraps(litellm.image_generation)
    def _image_generation_wrapper(self, *args: Any, **kwargs: Any) -> ImageResponse:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return self.original_litellm_funcs["image_generation"](*args, **kwargs)  # type:ignore
        with self._tracer.start_as_current_span(
            name="image_generation", attributes=dict(get_attributes_from_context())
        ) as span:
            _instrument_func_type_image_generation(span, kwargs)
            result = self.original_litellm_funcs["image_generation"](*args, **kwargs)
            _finalize_span(span, result)
        return result  # type:ignore

    @wraps(litellm.aimage_generation)
    async def _aimage_generation_wrapper(self, *args: Any, **kwargs: Any) -> ImageResponse:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return self.original_litellm_funcs["aimage_generation"](*args, **kwargs)  # type:ignore
        with self._tracer.start_as_current_span(
            name="aimage_generation", attributes=dict(get_attributes_from_context())
        ) as span:
            _instrument_func_type_image_generation(span, kwargs)
            result = await self.original_litellm_funcs["aimage_generation"](*args, **kwargs)
            _finalize_span(span, result)
        return result  # type:ignore

    def _set_wrapper_attr(self, func_wrapper: Any) -> None:
        func_wrapper.__func__.is_wrapper = True
