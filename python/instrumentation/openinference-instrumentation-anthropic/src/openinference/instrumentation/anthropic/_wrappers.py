from __future__ import annotations

import logging
from abc import ABC
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from contextvars import ContextVar, Token
from itertools import chain
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Type,
)

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.types import AttributeValue
from typing_extensions import assert_never
from wrapt.proxies import ObjectProxy

from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.instrumentation.anthropic._stream import (
    _MessagesStream,
    _RawStreamInterceptor,
    _Stream,
)
from openinference.instrumentation.anthropic._with_span import _WithSpan
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if TYPE_CHECKING:
    from pydantic import BaseModel

    from anthropic.lib.streaming import (
        AsyncMessageStreamManager,
        BetaAsyncMessageStreamManager,
        BetaMessageStreamManager,
        MessageStreamManager,
    )
    from anthropic.types import Message, MessageParam, ToolUnionParam, Usage


def _stop_on_exception(
    wrapped: Callable[..., Iterator[Tuple[str, Any]]],
) -> Callable[..., Iterator[Tuple[str, Any]]]:
    def wrapper(*args: Any, **kwargs: Any) -> Iterator[Tuple[str, Any]]:
        try:
            yield from wrapped(*args, **kwargs)
        except Exception:
            logger.exception("Failed to get attribute.")

    return wrapper


class _Params:
    def __init__(
        self,
        kwargs: Mapping[str, Any],
        get_attributes: Optional[
            Callable[[Mapping[str, Any]], Iterator[Tuple[str, AttributeValue]]]
        ] = None,
    ) -> None:
        self._get_attributes = get_attributes or (lambda kwargs: iter(kwargs.items()))
        self._kwargs = dict(kwargs)
        self._token: Optional[Token[Optional[_Params]]] = None
        self._updated = False  # prevent spurious updates

    def __iter__(self) -> Iterator[Tuple[str, AttributeValue]]:
        return self._get_attributes(dict(self._kwargs))

    def __enter__(self) -> _Params:
        self._token = _params.set(self)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        if self._token:
            _params.reset(self._token)

    def update(self, **kwargs: Any) -> None:
        if self._updated:
            return
        self._kwargs.update(kwargs)
        self._updated = True


_params: ContextVar[Optional[_Params]] = ContextVar("params", default=None)


class _TransformWrapper:
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        params = _params.get()
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY) or params is None:
            return wrapped(*args, **kwargs)
        ans = wrapped(*args, **kwargs)
        if isinstance(ans, Mapping):
            params.update(**ans)
        return ans


class _AsyncTransformWrapper:
    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        params = _params.get()
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY) or params is None:
            return await wrapped(*args, **kwargs)
        ans = await wrapped(*args, **kwargs)
        if isinstance(ans, Mapping):
            params.update(**ans)
        return ans


class _WithTracer(ABC):
    """
    Base class for wrappers that need a tracer.
    """

    def __init__(
        self,
        tracer: trace_api.Tracer,
        span_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer
        self._span_name = span_name

    @contextmanager
    def _start_as_current_span(
        self,
        attributes: Optional[Mapping[str, Any]] = None,
        params: AbstractContextManager[Iterable[Tuple[str, AttributeValue]]] = nullcontext(()),
    ) -> Iterator[_WithSpan]:
        try:
            span = self._tracer.start_span(
                name=self._span_name,
                record_exception=False,
                set_status_on_exception=False,
                attributes=attributes,
            )
        except Exception:
            span = INVALID_SPAN
        with ExitStack() as stack:
            stack.enter_context(
                trace_api.use_span(
                    span,
                    end_on_exit=False,
                    record_exception=False,
                    set_status_on_exception=False,
                )
            )
            yield _WithSpan(span=span, params=stack.enter_context(params))


@_stop_on_exception
def _get_attributes_from_completions_create(
    kwargs: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    yield from _get_llm_model_name_from_input(kwargs)
    invocation_parameters = dict(kwargs)
    invocation_parameters.pop("extra_headers", None)
    invocation_parameters.pop("model", None)
    if prompt := invocation_parameters.pop("prompt", None):
        yield from _get_llm_prompts(prompt)
    if isinstance(tools := invocation_parameters.pop("tools", None), Iterable):
        yield from _get_llm_tools(tools)
    yield LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_parameters)


class _CompletionsWrapper(_WithTracer):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    __slots__ = "_response_accumulator"

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        with self._start_as_current_span(
            params=_Params(kwargs, get_attributes=_get_attributes_from_completions_create),
            attributes=dict(
                chain(
                    get_attributes_from_context(),
                    _get_llm_provider(),
                    _get_llm_system(),
                    _get_llm_span_kind(),
                    _get_inputs(kwargs),
                )
            ),
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                span.finish_tracing()
                raise
        streaming = kwargs.get("stream", False)
        if streaming:
            return _Stream(response, span)
        else:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attributes(dict(_get_outputs(response)))
            span.finish_tracing()
            return response


class _AsyncCompletionsWrapper(_WithTracer):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        with self._start_as_current_span(
            params=_Params(kwargs, get_attributes=_get_attributes_from_completions_create),
            attributes=dict(
                chain(
                    get_attributes_from_context(),
                    _get_llm_provider(),
                    _get_llm_system(),
                    _get_llm_span_kind(),
                    _get_inputs(kwargs),
                )
            ),
        ) as span:
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                span.finish_tracing()
                raise
        streaming = kwargs.get("stream", False)
        if streaming:
            return _Stream(response, span)
        else:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attributes(dict(_get_outputs(response)))
            span.finish_tracing()
            return response


@_stop_on_exception
def _get_attributes_from_messages_create(
    kwargs: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    yield from _get_llm_model_name_from_input(kwargs)
    invocation_parameters = dict(kwargs)
    invocation_parameters.pop("extra_headers", None)
    invocation_parameters.pop("model", None)
    invocation_parameters.pop("output_format", None)
    if isinstance(messages := invocation_parameters.pop("messages", None), Iterable):
        yield from _get_llm_input_messages(messages)
    if isinstance(tools := invocation_parameters.pop("tools", None), Iterable):
        yield from _get_llm_tools(tools)
    yield LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_parameters)


class _MessagesWrapper(_WithTracer):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        with self._start_as_current_span(
            params=_Params(kwargs, get_attributes=_get_attributes_from_messages_create),
            attributes=dict(
                chain(
                    get_attributes_from_context(),
                    _get_llm_provider(),
                    _get_llm_system(),
                    _get_llm_span_kind(),
                    _get_inputs(kwargs),
                )
            ),
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                span.finish_tracing()
                raise
        streaming = kwargs.get("stream", False)
        if streaming:
            return _MessagesStream(response, span)
        else:
            span.finish_tracing(
                status=trace_api.Status(trace_api.StatusCode.OK),
                extra_attributes=dict(
                    chain(
                        _get_llm_model_name_from_response(response),
                        _get_output_messages(response),
                        _get_llm_token_counts(response.usage),
                        _get_outputs(response),
                    )
                ),
            )
            return response


class _AsyncMessagesWrapper(_WithTracer):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        with self._start_as_current_span(
            params=_Params(kwargs, get_attributes=_get_attributes_from_messages_create),
            attributes=dict(
                chain(
                    get_attributes_from_context(),
                    _get_llm_provider(),
                    _get_llm_system(),
                    _get_llm_span_kind(),
                    _get_inputs(kwargs),
                )
            ),
        ) as span:
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                span.finish_tracing()
                raise
        streaming = kwargs.get("stream", False)
        if streaming:
            return _MessagesStream(response, span)
        else:
            span.finish_tracing(
                status=trace_api.Status(trace_api.StatusCode.OK),
                extra_attributes=dict(
                    chain(
                        _get_llm_model_name_from_response(response),
                        _get_output_messages(response),
                        _get_llm_token_counts(response.usage),
                        _get_outputs(response),
                    )
                ),
            )
            return response


class _MessagesStreamWrapper(_WithTracer):
    # The manager proxy class to instantiate. Each must be a differently-named
    # proxy so Python name-mangling resolves __api_request against the correct
    # SDK manager class (e.g. _MessageStreamManager strips to
    # MessageStreamManager, matching the SDK attribute _MessageStreamManager__api_request).
    def __init__(
        self,
        tracer: trace_api.Tracer,
        span_name: str,
        manager_class: "Type[_MessageStreamManager]",
    ) -> None:
        super().__init__(tracer=tracer, span_name=span_name)
        self._manager_class = manager_class

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        with self._start_as_current_span(
            params=_Params(kwargs, get_attributes=_get_attributes_from_messages_create),
            attributes=dict(
                chain(
                    get_attributes_from_context(),
                    _get_llm_provider(),
                    _get_llm_system(),
                    _get_llm_span_kind(),
                    _get_inputs(kwargs),
                )
            ),
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR))
                span.record_exception(exception)
                span.finish_tracing()
                raise
        return self._manager_class(response, span)


class _AsyncMessagesStreamWrapper(_WithTracer):
    def __init__(
        self,
        tracer: trace_api.Tracer,
        span_name: str,
        manager_class: "Type[_AsyncMessageStreamManager]",
    ) -> None:
        super().__init__(tracer=tracer, span_name=span_name)
        self._manager_class = manager_class

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        with self._start_as_current_span(
            params=_Params(kwargs, get_attributes=_get_attributes_from_messages_create),
            attributes=dict(
                chain(
                    get_attributes_from_context(),
                    _get_llm_provider(),
                    _get_llm_system(),
                    _get_llm_span_kind(),
                    _get_inputs(kwargs),
                )
            ),
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR))
                span.record_exception(exception)
                span.finish_tracing()
                raise
        return self._manager_class(response, span)


# Sync stream manager proxies.
# These proxy the SDK's MessageStreamManager, intercept the raw HTTP stream for
# span accumulation via _RawStreamInterceptor, and return the real MessageStream
# so callers have full access to .text_stream, .get_final_message(), etc.


class _MessageStreamManager(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    __slots__ = ("_self_with_span", "_self_interceptor", "_self_message_stream")

    def __init__(
        self,
        manager: "MessageStreamManager",
        with_span: _WithSpan,
    ) -> None:
        super().__init__(manager)
        self._self_with_span = with_span
        self._self_interceptor: Optional[_RawStreamInterceptor] = None
        self._self_message_stream: Any = None

    def __enter__(self) -> Any:
        message_stream = self.__wrapped__.__enter__()
        interceptor = _RawStreamInterceptor(
            message_stream._raw_stream, self._self_with_span, message_stream
        )
        message_stream._raw_stream = interceptor
        self._self_interceptor = interceptor
        self._self_message_stream = message_stream
        return message_stream

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        if self._self_interceptor is not None:
            self._self_interceptor._finish_tracing()
        if self._self_message_stream is not None:
            self._self_message_stream.close()


class _BetaMessageStreamManager(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    __slots__ = ("_self_with_span", "_self_interceptor", "_self_message_stream")

    def __init__(
        self,
        manager: "BetaMessageStreamManager",
        with_span: _WithSpan,
    ) -> None:
        super().__init__(manager)
        self._self_with_span = with_span
        self._self_interceptor: Optional[_RawStreamInterceptor] = None
        self._self_message_stream: Any = None

    def __enter__(self) -> Any:
        message_stream = self.__wrapped__.__enter__()
        interceptor = _RawStreamInterceptor(
            message_stream._raw_stream, self._self_with_span, message_stream
        )
        message_stream._raw_stream = interceptor
        self._self_interceptor = interceptor
        self._self_message_stream = message_stream
        return message_stream

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        if self._self_interceptor is not None:
            self._self_interceptor._finish_tracing()
        if self._self_message_stream is not None:
            self._self_message_stream.close()


# Async stream manager proxies.


class _AsyncMessageStreamManager(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    __slots__ = ("_self_with_span", "_self_interceptor", "_self_message_stream")

    def __init__(
        self,
        manager: "AsyncMessageStreamManager",
        with_span: _WithSpan,
    ) -> None:
        super().__init__(manager)
        self._self_with_span = with_span
        self._self_interceptor: Optional[_RawStreamInterceptor] = None
        self._self_message_stream: Any = None

    async def __aenter__(self) -> Any:
        message_stream = await self.__wrapped__.__aenter__()
        interceptor = _RawStreamInterceptor(
            message_stream._raw_stream, self._self_with_span, message_stream
        )
        message_stream._raw_stream = interceptor
        self._self_interceptor = interceptor
        self._self_message_stream = message_stream
        return message_stream

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        if self._self_interceptor is not None:
            self._self_interceptor._finish_tracing()
        if self._self_message_stream is not None:
            await self._self_message_stream.close()


class _BetaAsyncMessageStreamManager(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    __slots__ = ("_self_with_span", "_self_interceptor", "_self_message_stream")

    def __init__(
        self,
        manager: "BetaAsyncMessageStreamManager",
        with_span: _WithSpan,
    ) -> None:
        super().__init__(manager)
        self._self_with_span = with_span
        self._self_interceptor: Optional[_RawStreamInterceptor] = None
        self._self_message_stream: Any = None

    async def __aenter__(self) -> Any:
        message_stream = await self.__wrapped__.__aenter__()
        interceptor = _RawStreamInterceptor(
            message_stream._raw_stream, self._self_with_span, message_stream
        )
        message_stream._raw_stream = interceptor
        self._self_interceptor = interceptor
        self._self_message_stream = message_stream
        return message_stream

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        if self._self_interceptor is not None:
            self._self_interceptor._finish_tracing()
        if self._self_message_stream is not None:
            await self._self_message_stream.close()


@_stop_on_exception
def _get_inputs(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    yield INPUT_VALUE, safe_json_dumps(arguments)
    yield INPUT_MIME_TYPE, JSON


@_stop_on_exception
def _get_outputs(response: "BaseModel") -> Iterator[Tuple[str, Any]]:
    yield OUTPUT_VALUE, response.model_dump_json()
    yield OUTPUT_MIME_TYPE, JSON


@_stop_on_exception
def _get_llm_tools(tools: Iterable[ToolUnionParam]) -> Iterator[Tuple[str, Any]]:
    for tool_index, tool_schema in enumerate(tools):
        yield f"{LLM_TOOLS}.{tool_index}.{TOOL_JSON_SCHEMA}", safe_json_dumps(tool_schema)


@_stop_on_exception
def _get_llm_span_kind() -> Iterator[Tuple[str, Any]]:
    yield OPENINFERENCE_SPAN_KIND, LLM


@_stop_on_exception
def _get_llm_provider() -> Iterator[Tuple[str, Any]]:
    yield LLM_PROVIDER, LLM_PROVIDER_ANTHROPIC


@_stop_on_exception
def _get_llm_system() -> Iterator[Tuple[str, Any]]:
    yield LLM_SYSTEM, LLM_SYSTEM_ANTHROPIC


@_stop_on_exception
def _get_llm_token_counts(usage: "Usage") -> Iterator[Tuple[str, Any]]:
    # See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#tracking-cache-performance
    # cache_creation_input_tokens: Number of tokens written to the cache when creating a new entry.
    # cache_read_input_tokens: Number of tokens retrieved from the cache for this request.
    # input_tokens: Number of input tokens which were not read from or used to create a cache.
    if prompt_tokens := (
        usage.input_tokens
        + (usage.cache_creation_input_tokens or 0)
        + (usage.cache_read_input_tokens or 0)
    ):
        yield LLM_TOKEN_COUNT_PROMPT, prompt_tokens
    if usage.output_tokens:
        yield LLM_TOKEN_COUNT_COMPLETION, usage.output_tokens
    if usage.cache_read_input_tokens:
        yield LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ, usage.cache_read_input_tokens
    if usage.cache_creation_input_tokens:
        yield LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE, usage.cache_creation_input_tokens


@_stop_on_exception
def _get_llm_model_name_from_input(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    if model_name := arguments.get("model"):
        yield LLM_MODEL_NAME, model_name


@_stop_on_exception
def _get_llm_model_name_from_response(message: "Message") -> Iterator[Tuple[str, Any]]:
    if model_name := message.model:
        yield LLM_MODEL_NAME, model_name


@_stop_on_exception
def _get_llm_prompts(prompt: str) -> Iterator[Tuple[str, Any]]:
    yield LLM_PROMPTS, [prompt]


@_stop_on_exception
def _get_llm_input_messages(messages: Iterable[MessageParam]) -> Iterator[Tuple[str, Any]]:
    """
    Extracts the messages from the chat response
    """

    for i, message in enumerate(messages):
        tool_index = 0
        if role := message["role"]:
            yield f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_ROLE}", role
        if content := message["content"]:
            if isinstance(content, str):
                yield f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_CONTENT}", content
                continue
            for j, block in enumerate(content):
                if isinstance(block, dict):
                    if block["type"] == "text":
                        yield f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_CONTENT}", block["text"]
                    elif block["type"] == "tool_use":
                        yield (
                            f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_TOOL_CALLS}.{tool_index}.{TOOL_CALL_ID}",
                            block["id"],
                        )
                        yield (
                            f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_TOOL_CALLS}.{tool_index}.{TOOL_CALL_FUNCTION_NAME}",
                            block["name"],
                        )
                        yield (
                            f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_TOOL_CALLS}.{tool_index}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                            safe_json_dumps(block["input"]),
                        )
                        tool_index += 1
                    elif block["type"] == "tool_result":
                        yield (
                            f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_TOOL_CALL_ID}",
                            block["tool_use_id"],
                        )
                        if (tool_result_content := block.get("content")) is not None:
                            yield (
                                f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_CONTENT}",
                                tool_result_content
                                if isinstance(tool_result_content, str)
                                else safe_json_dumps(tool_result_content),
                            )
                    elif block["type"] == "image":
                        if source := block["source"]:
                            image_data = f"data:{source.get('media_type')};{source.get('type')}"
                            image_data = f"{image_data},{source.get('data')}"
                            prefix = f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_CONTENTS}.{j}"
                            yield f"{prefix}.{MESSAGE_CONTENT_TYPE}", "image"
                            yield f"{prefix}.{MESSAGE_CONTENT_IMAGE}.{IMAGE_URL}", image_data
                    elif block["type"] == "document":
                        pass
                    elif block["type"] == "search_result":
                        pass
                    elif block["type"] == "thinking":
                        pass
                    elif block["type"] == "redacted_thinking":
                        pass
                    elif block["type"] == "server_tool_use":
                        pass
                    elif block["type"] == "web_search_tool_result":
                        pass
                    elif block["type"] == "web_fetch_tool_result":
                        pass
                    elif block["type"] == "code_execution_tool_result":
                        pass
                    elif block["type"] == "bash_code_execution_tool_result":
                        pass
                    elif block["type"] == "text_editor_code_execution_tool_result":
                        pass
                    elif block["type"] == "tool_search_tool_result":
                        pass
                    elif block["type"] == "container_upload":
                        pass
                    elif TYPE_CHECKING:
                        assert_never(block)
                else:
                    if block.type == "text":
                        yield f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_CONTENT}", block.text
                    elif block.type == "tool_use":
                        if tool_call_id := block.id:
                            yield (
                                f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_TOOL_CALLS}.{tool_index}.{TOOL_CALL_ID}",
                                tool_call_id,
                            )
                        yield (
                            f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_TOOL_CALLS}.{tool_index}.{TOOL_CALL_FUNCTION_NAME}",
                            block.name,
                        )
                        yield (
                            f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_TOOL_CALLS}.{tool_index}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                            safe_json_dumps(block.input),
                        )
                        tool_index += 1
                    elif block.type == "thinking":
                        pass
                    elif block.type == "redacted_thinking":
                        pass
                    elif block.type == "server_tool_use":
                        pass
                    elif block.type == "web_search_tool_result":
                        pass
                    elif block.type == "web_fetch_tool_result":
                        pass
                    elif block.type == "code_execution_tool_result":
                        pass
                    elif block.type == "bash_code_execution_tool_result":
                        pass
                    elif block.type == "text_editor_code_execution_tool_result":
                        pass
                    elif block.type == "tool_search_tool_result":
                        pass
                    elif block.type == "container_upload":
                        pass
                    elif TYPE_CHECKING:
                        assert_never(block)


@_stop_on_exception
def _get_output_messages(response: Message) -> Iterator[Tuple[str, Any]]:
    """
    Extracts the tool call information from the response
    """
    yield f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}", response.role
    tool_index = 0
    for block in response.content:
        if block.type == "text":
            yield f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}", block.text
        elif block.type == "tool_use":
            yield (
                f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.{tool_index}.{TOOL_CALL_FUNCTION_NAME}",
                block.name,
            )
            yield (
                f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.{tool_index}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                safe_json_dumps(block.input),
            )
            tool_index += 1
        elif block.type == "thinking":
            pass
        elif block.type == "redacted_thinking":
            pass
        elif block.type == "server_tool_use":
            pass
        elif block.type == "web_search_tool_result":
            pass
        elif block.type == "web_fetch_tool_result":
            pass
        elif block.type == "code_execution_tool_result":
            pass
        elif block.type == "bash_code_execution_tool_result":
            pass
        elif block.type == "text_editor_code_execution_tool_result":
            pass
        elif block.type == "tool_search_tool_result":
            pass
        elif block.type == "container_upload":
            pass
        elif TYPE_CHECKING:
            assert_never(block)


CHAIN = OpenInferenceSpanKindValues.CHAIN.value
EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING.value
LLM = OpenInferenceSpanKindValues.LLM.value
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER.value

JSON = OpenInferenceMimeTypeValues.JSON.value
TEXT = OpenInferenceMimeTypeValues.TEXT.value

DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
DOCUMENT_SCORE = DocumentAttributes.DOCUMENT_SCORE
DOCUMENT_METADATA = DocumentAttributes.DOCUMENT_METADATA
EMBEDDING_EMBEDDINGS = SpanAttributes.EMBEDDING_EMBEDDINGS
EMBEDDING_MODEL_NAME = SpanAttributes.EMBEDDING_MODEL_NAME
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES
LLM_PROMPT_TEMPLATE_VERSION = SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE = (
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE
)
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOOLS = SpanAttributes.LLM_TOOLS
IMAGE_URL = ImageAttributes.IMAGE_URL
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
MESSAGE_TOOL_CALL_ID = MessageAttributes.MESSAGE_TOOL_CALL_ID
METADATA = SpanAttributes.METADATA
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
SESSION_ID = SpanAttributes.SESSION_ID
TAG_TAGS = SpanAttributes.TAG_TAGS
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA
USER_ID = SpanAttributes.USER_ID
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
LLM_PROVIDER_ANTHROPIC = OpenInferenceLLMProviderValues.ANTHROPIC.value
LLM_SYSTEM_ANTHROPIC = OpenInferenceLLMSystemValues.ANTHROPIC.value
