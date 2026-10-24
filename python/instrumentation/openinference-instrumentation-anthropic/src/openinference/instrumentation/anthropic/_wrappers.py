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
from weakref import WeakSet

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.types import AttributeValue
from wrapt import ObjectProxy

from openinference.instrumentation import (
    TraceConfig,
    get_attributes_from_context,
    safe_json_dumps,
)
from openinference.instrumentation.anthropic._content import (
    _attributes_or_nothing,
    _dump_json,
    _get,
    _get_message_contents_attributes,
    _get_metadata_attributes,
    _get_offered_tools,
    _get_output_message_attributes,
    _get_request_params,
    _hides_input_images,
    _require,
    _to_json,
    _to_jsonable,
)
from openinference.instrumentation.anthropic._stream import (
    _MessagesStream,
    _RawStreamInterceptor,
)
from openinference.instrumentation.anthropic._utils import _get_token_counts
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
    from anthropic.types import Message, MessageParam, TextBlockParam, ToolUnionParam, Usage
    from anthropic.types.message_create_params import MessageCreateParamsBase


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

    @property
    def request_messages(self) -> Any:
        return self._kwargs.get("messages")

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

    def update(self, kwargs: Mapping[Any, Any]) -> None:
        # a mapping, not keyword arguments: a request body can have any keys, e.g. "self" or
        # non-strings from extra_body
        if self._updated:
            return
        self._kwargs.update(kwargs)
        self._updated = True


_params: ContextVar[Optional[_Params]] = ContextVar("params", default=None)


class _PrepareRequestDataWrapper:
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
        if kwargs.get("location") == "body" and isinstance(ans, Mapping):
            try:
                params.update(ans)
            except Exception:
                logger.exception("Failed to record the prepared request body")
        return ans


class _AsyncPrepareRequestDataWrapper:
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
        if kwargs.get("location") == "body" and isinstance(ans, Mapping):
            try:
                params.update(ans)
            except Exception:
                logger.exception("Failed to record the prepared request body")
        return ans


class _WithTracer(ABC):
    """
    Base class for wrappers that need a tracer.
    """

    # every wrapper made, so uninstrument() can turn off the ones the SDK
    # keeps in cached accessors like client.messages.with_raw_response
    _instances: WeakSet[_WithTracer] = WeakSet()

    def __init__(
        self,
        tracer: trace_api.Tracer,
        span_name: str,
        *args: Any,
        config: Optional[TraceConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer
        self._span_name = span_name
        # the trace config, for the attributes that the tracer's masking does not cover
        self._config = config or TraceConfig()
        self._active = True
        _WithTracer._instances.add(self)

    def _get_attributes_from_request(
        self, kwargs: Mapping[str, Any]
    ) -> Iterator[Tuple[str, AttributeValue]]:
        return _get_attributes_from_messages_create(kwargs, config=self._config)

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
            yield _WithSpan(
                span=span,
                params=stack.enter_context(params),
                context_metadata=(attributes or {}).get(METADATA),
                config=self._config,
            )


@_stop_on_exception
def _get_attributes_from_messages_create(
    kwargs: MessageCreateParamsBase,
    config: Optional[TraceConfig] = None,
) -> Iterator[Tuple[str, AttributeValue]]:
    yield from _get_llm_model_name_from_input(kwargs)
    invocation_parameters = _get_request_params(kwargs)
    invocation_parameters.pop("extra_headers", None)
    invocation_parameters.pop("model", None)
    invocation_parameters.pop("output_format", None)
    system = invocation_parameters.pop("system", None)
    messages = invocation_parameters.pop("messages", None)
    # a list, because the messages are read twice
    messages = list(messages) if isinstance(messages, Iterable) else []
    yield from _get_llm_input_messages(messages, system, _hides_input_images(config))
    tools = invocation_parameters.pop("tools", None)
    tools = _get_offered_tools(tools if isinstance(tools, Iterable) else (), messages)
    if tools:
        yield from _get_llm_tools(tools)
    yield LLM_INVOCATION_PARAMETERS, _to_json(invocation_parameters)


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
        if not self._active or context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        with self._start_as_current_span(
            params=_Params(kwargs, get_attributes=self._get_attributes_from_request),
            attributes=dict(
                chain(
                    get_attributes_from_context(),
                    _get_llm_provider(instance),
                    _get_llm_system(),
                    _get_llm_span_kind(),
                    _get_inputs(kwargs),
                )
            ),
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except BaseException as exception:
                # BaseException, so that e.g. KeyboardInterrupt also finishes the span
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                span.finish_tracing()
                raise
        if _is_api_response(response):
            _finish_api_response_tracing(span, response, streaming=kwargs.get("stream", False))
            return response
        streaming = kwargs.get("stream", False)
        if streaming:
            return _MessagesStream(response, span, is_beta=self._span_name.startswith("beta."))
        else:
            _finish_message_tracing(span, response)
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
        if not self._active or context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        with self._start_as_current_span(
            params=_Params(kwargs, get_attributes=self._get_attributes_from_request),
            attributes=dict(
                chain(
                    get_attributes_from_context(),
                    _get_llm_provider(instance),
                    _get_llm_system(),
                    _get_llm_span_kind(),
                    _get_inputs(kwargs),
                )
            ),
        ) as span:
            try:
                response = await wrapped(*args, **kwargs)
            except BaseException as exception:
                # BaseException, so that e.g. a cancelled task's CancelledError also finishes
                # the span
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                span.finish_tracing()
                raise
        if _is_api_response(response):
            await _async_finish_api_response_tracing(
                span, response, streaming=kwargs.get("stream", False)
            )
            return response
        streaming = kwargs.get("stream", False)
        if streaming:
            return _MessagesStream(response, span, is_beta=self._span_name.startswith("beta."))
        else:
            _finish_message_tracing(span, response)
            return response


def _finish_message_tracing(span: _WithSpan, message: Any) -> None:
    span.finish_tracing(
        status=trace_api.Status(trace_api.StatusCode.OK),
        extra_attributes=dict(
            chain(
                _get_llm_model_name_from_response(message),
                _get_output_messages(message),
                _get_metadata(message, span),
                _get_llm_token_counts_from_response(message),
                _get_outputs(message),
            )
        ),
    )


class _CountTokensWrapper(_WithTracer):
    """
    Wrapper for messages.count_tokens, which counts the input tokens of a request without
    sending it to the model, so its span has the request's input and the count as its output.
    """

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if not self._active or context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        with self._start_as_current_span(
            params=_Params(kwargs, get_attributes=self._get_attributes_from_request),
            attributes=dict(
                chain(
                    _get_llm_provider(instance),
                    _get_llm_system(),
                    _get_llm_span_kind(),
                    _get_inputs(kwargs),
                )
            ),
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except BaseException as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                span.finish_tracing()
                raise
        if _is_api_response(response):
            _finish_api_response_tracing(
                span, response, streaming=False, finish=_finish_count_tokens_tracing
            )
        else:
            _finish_count_tokens_tracing(span, response)
        return response


class _AsyncCountTokensWrapper(_WithTracer):
    """
    See _CountTokensWrapper.
    """

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if not self._active or context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        with self._start_as_current_span(
            params=_Params(kwargs, get_attributes=self._get_attributes_from_request),
            attributes=dict(
                chain(
                    _get_llm_provider(instance),
                    _get_llm_system(),
                    _get_llm_span_kind(),
                    _get_inputs(kwargs),
                )
            ),
        ) as span:
            try:
                response = await wrapped(*args, **kwargs)
            except BaseException as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                span.finish_tracing()
                raise
        if _is_api_response(response):
            await _async_finish_api_response_tracing(
                span, response, streaming=False, finish=_finish_count_tokens_tracing
            )
        else:
            _finish_count_tokens_tracing(span, response)
        return response


def _finish_count_tokens_tracing(span: _WithSpan, response: Any) -> None:
    span.finish_tracing(
        status=trace_api.Status(trace_api.StatusCode.OK),
        extra_attributes=dict(_get_count_tokens_outputs(response)),
    )


@_stop_on_exception
def _get_count_tokens_outputs(response: Any) -> Iterator[Tuple[str, Any]]:
    # The response, e.g. the count and, on beta, the context management's effect on it, with the
    # API's field names. The count is not recorded as a token count, since nothing is generated
    # or billed.
    yield OUTPUT_VALUE, safe_json_dumps(_to_jsonable(response))
    yield OUTPUT_MIME_TYPE, JSON


def _is_api_response(response: Any) -> bool:
    """
    messages.with_raw_response and messages.with_streaming_response return the HTTP response,
    wrapped in an APIResponse, instead of the message.
    """
    from anthropic import APIResponse, AsyncAPIResponse

    return isinstance(response, (APIResponse, AsyncAPIResponse))


def _parsing_runs_caller_code(response: Any) -> bool:
    """
    Parsing applies the request's post_parser, which is caller code: messages.parse() sets one
    to validate the response against the caller's output_format, and middleware can set one on
    any request. Caller code must run only when the caller parses a raw response itself.
    """
    from anthropic import NotGiven

    options = getattr(response, "_options", None)
    return not isinstance(getattr(options, "post_parser", None), NotGiven)


def _finish_api_response_tracing(
    span: _WithSpan,
    response: Any,
    streaming: bool,
    finish: Callable[[_WithSpan, Any], None] = _finish_message_tracing,
) -> None:
    """
    The output attributes are recorded only if the body has been read and parsing runs no caller
    code; the response caches the parsed message for the caller's own parse(). An unread body
    leaves the outcome unknown, e.g. a stream can still end in an error event, so the status is
    left unset. An event stream counts as unread even if a transport buffered its body.
    """
    if streaming or not response.is_closed:
        span.finish_tracing()
        return
    if _parsing_runs_caller_code(response):
        span.finish_tracing(status=trace_api.Status(trace_api.StatusCode.OK))
        return
    try:
        message = response.parse()
    except Exception:
        logger.exception("Failed to parse the raw response")
        span.finish_tracing(status=trace_api.Status(trace_api.StatusCode.OK))
        return
    finish(span, message)


async def _async_finish_api_response_tracing(
    span: _WithSpan,
    response: Any,
    streaming: bool,
    finish: Callable[[_WithSpan, Any], None] = _finish_message_tracing,
) -> None:
    """
    See _finish_api_response_tracing.
    """
    if streaming or not response.is_closed:
        span.finish_tracing()
        return
    if _parsing_runs_caller_code(response):
        span.finish_tracing(status=trace_api.Status(trace_api.StatusCode.OK))
        return
    try:
        message = await response.parse()
    except Exception:
        logger.exception("Failed to parse the raw response")
        span.finish_tracing(status=trace_api.Status(trace_api.StatusCode.OK))
        return
    finish(span, message)


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
        config: Optional[TraceConfig] = None,
    ) -> None:
        super().__init__(tracer=tracer, span_name=span_name, config=config)
        self._manager_class = manager_class

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if not self._active or context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        params = _Params(kwargs, get_attributes=self._get_attributes_from_request)
        with self._start_as_current_span(
            params=params,
            attributes=dict(
                chain(
                    get_attributes_from_context(),
                    _get_llm_provider(instance),
                    _get_llm_system(),
                    _get_llm_span_kind(),
                    _get_inputs(kwargs),
                )
            ),
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except BaseException as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR))
                span.record_exception(exception)
                span.finish_tracing()
                raise
        return self._manager_class(response, span, params)


class _AsyncMessagesStreamWrapper(_WithTracer):
    def __init__(
        self,
        tracer: trace_api.Tracer,
        span_name: str,
        manager_class: "Type[_AsyncMessageStreamManager]",
        config: Optional[TraceConfig] = None,
    ) -> None:
        super().__init__(tracer=tracer, span_name=span_name, config=config)
        self._manager_class = manager_class

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if not self._active or context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        params = _Params(kwargs, get_attributes=self._get_attributes_from_request)
        with self._start_as_current_span(
            params=params,
            attributes=dict(
                chain(
                    get_attributes_from_context(),
                    _get_llm_provider(instance),
                    _get_llm_system(),
                    _get_llm_span_kind(),
                    _get_inputs(kwargs),
                )
            ),
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except BaseException as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR))
                span.record_exception(exception)
                span.finish_tracing()
                raise
        return self._manager_class(response, span, params)


# Sync stream manager proxies.
# These proxy the SDK's MessageStreamManager, intercept the raw HTTP stream for
# span accumulation via _RawStreamInterceptor, and return the real MessageStream
# so callers have full access to .text_stream, .get_final_message(), etc.


class _MessageStreamManager(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    __slots__ = ("_self_with_span", "_self_params", "_self_interceptor", "_self_message_stream")

    def __init__(
        self,
        manager: "MessageStreamManager",
        with_span: _WithSpan,
        params: _Params,
    ) -> None:
        super().__init__(manager)
        self._self_with_span = with_span
        self._self_params = params
        self._self_interceptor: Optional[_RawStreamInterceptor] = None
        self._self_message_stream: Any = None

    def __enter__(self) -> Any:
        try:
            # the request is sent here, after stream() has returned, and anthropic>=1.8.0
            # prepares its body here, so the params are made current again for the body
            # preparation wrapper
            with self._self_params:
                message_stream = self.__wrapped__.__enter__()
        except BaseException as exception:
            # a failed request never reaches __exit__, so the span has to be finished here
            self._self_with_span.set_status(
                trace_api.Status(trace_api.StatusCode.ERROR, str(exception))
            )
            self._self_with_span.record_exception(exception)
            self._self_with_span.finish_tracing()
            raise
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
    __slots__ = ("_self_with_span", "_self_params", "_self_interceptor", "_self_message_stream")

    def __init__(
        self,
        manager: "BetaMessageStreamManager",
        with_span: _WithSpan,
        params: _Params,
    ) -> None:
        super().__init__(manager)
        self._self_with_span = with_span
        self._self_params = params
        self._self_interceptor: Optional[_RawStreamInterceptor] = None
        self._self_message_stream: Any = None

    def __enter__(self) -> Any:
        try:
            # the request is sent here, after stream() has returned, and anthropic>=1.8.0
            # prepares its body here, so the params are made current again for the body
            # preparation wrapper
            with self._self_params:
                message_stream = self.__wrapped__.__enter__()
        except BaseException as exception:
            # a failed request never reaches __exit__, so the span has to be finished here
            self._self_with_span.set_status(
                trace_api.Status(trace_api.StatusCode.ERROR, str(exception))
            )
            self._self_with_span.record_exception(exception)
            self._self_with_span.finish_tracing()
            raise
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
    __slots__ = ("_self_with_span", "_self_params", "_self_interceptor", "_self_message_stream")

    def __init__(
        self,
        manager: "AsyncMessageStreamManager",
        with_span: _WithSpan,
        params: _Params,
    ) -> None:
        super().__init__(manager)
        self._self_with_span = with_span
        self._self_params = params
        self._self_interceptor: Optional[_RawStreamInterceptor] = None
        self._self_message_stream: Any = None

    async def __aenter__(self) -> Any:
        try:
            # the request is sent here, after stream() has returned, and anthropic>=1.8.0
            # prepares its body here, so the params are made current again for the body
            # preparation wrapper
            with self._self_params:
                message_stream = await self.__wrapped__.__aenter__()
        except BaseException as exception:
            # a failed request never reaches __exit__, so the span has to be finished here
            self._self_with_span.set_status(
                trace_api.Status(trace_api.StatusCode.ERROR, str(exception))
            )
            self._self_with_span.record_exception(exception)
            self._self_with_span.finish_tracing()
            raise
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
    __slots__ = ("_self_with_span", "_self_params", "_self_interceptor", "_self_message_stream")

    def __init__(
        self,
        manager: "BetaAsyncMessageStreamManager",
        with_span: _WithSpan,
        params: _Params,
    ) -> None:
        super().__init__(manager)
        self._self_with_span = with_span
        self._self_params = params
        self._self_interceptor: Optional[_RawStreamInterceptor] = None
        self._self_message_stream: Any = None

    async def __aenter__(self) -> Any:
        try:
            # the request is sent here, after stream() has returned, and anthropic>=1.8.0
            # prepares its body here, so the params are made current again for the body
            # preparation wrapper
            with self._self_params:
                message_stream = await self.__wrapped__.__aenter__()
        except BaseException as exception:
            # a failed request never reaches __exit__, so the span has to be finished here
            self._self_with_span.set_status(
                trace_api.Status(trace_api.StatusCode.ERROR, str(exception))
            )
            self._self_with_span.record_exception(exception)
            self._self_with_span.finish_tracing()
            raise
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
    # as JSON, so e.g. response blocks passed back as input are dumped rather than repr'd
    yield INPUT_VALUE, safe_json_dumps(_to_jsonable(_get_request_params(arguments)))
    yield INPUT_MIME_TYPE, JSON


@_stop_on_exception
def _get_outputs(response: "BaseModel") -> Iterator[Tuple[str, Any]]:
    yield OUTPUT_VALUE, _dump_json(response)
    yield OUTPUT_MIME_TYPE, JSON


@_stop_on_exception
def _get_llm_tools(tools: Iterable[ToolUnionParam]) -> Iterator[Tuple[str, Any]]:
    for tool_index, tool_schema in enumerate(tools):
        yield f"{LLM_TOOLS}.{tool_index}.{TOOL_JSON_SCHEMA}", safe_json_dumps(tool_schema)
        # `tool.name` and `tool.description` are separate spec attributes
        # (spec/tool_calling.md) and every Anthropic tool param carries them at the top
        # level, so consumers should not have to parse the schema blob to get them.
        if not isinstance(tool_schema, Mapping):
            continue
        if isinstance(name := tool_schema.get("name"), str) and name:
            yield f"{LLM_TOOLS}.{tool_index}.{TOOL_NAME}", name
        if isinstance(description := tool_schema.get("description"), str) and description:
            yield f"{LLM_TOOLS}.{tool_index}.{TOOL_DESCRIPTION}", description


@_stop_on_exception
def _get_llm_span_kind() -> Iterator[Tuple[str, Any]]:
    yield OPENINFERENCE_SPAN_KIND, LLM


# The base classes of the clients of the platforms that host Claude, by name, so that none of
# their modules, some of which need optional dependencies, has to be imported to check them.
_PLATFORM_CLIENT_PROVIDERS = {
    "BaseBedrockClient": OpenInferenceLLMProviderValues.AWS.value,
    "BaseMantleClient": OpenInferenceLLMProviderValues.AWS.value,  # Bedrock Mantle
    "AnthropicAWS": OpenInferenceLLMProviderValues.AWS.value,  # Claude Platform on AWS
    "AsyncAnthropicAWS": OpenInferenceLLMProviderValues.AWS.value,
    "BaseVertexClient": OpenInferenceLLMProviderValues.GOOGLE.value,
    "BaseGoogleCloudClient": OpenInferenceLLMProviderValues.GOOGLE.value,  # on Google Cloud
    "BaseFoundryClient": OpenInferenceLLMProviderValues.AZURE.value,  # Microsoft Foundry
}


@_stop_on_exception
def _get_llm_provider(instance: Any = None) -> Iterator[Tuple[str, Any]]:
    """
    The provider hosting the model, from the client the resource `instance` belongs to.
    """
    client = getattr(instance, "_client", None)
    for cls in type(client).__mro__:
        if (provider := _PLATFORM_CLIENT_PROVIDERS.get(cls.__name__)) and cls.__module__.startswith(
            "anthropic."
        ):
            yield LLM_PROVIDER, provider
            return
    yield LLM_PROVIDER, LLM_PROVIDER_ANTHROPIC


@_stop_on_exception
def _get_llm_system() -> Iterator[Tuple[str, Any]]:
    yield LLM_SYSTEM, LLM_SYSTEM_ANTHROPIC


@_stop_on_exception
def _get_llm_token_counts(usage: "Usage") -> Iterator[Tuple[str, Any]]:
    yield from _get_token_counts(usage)


@_stop_on_exception
def _get_llm_token_counts_from_response(message: "Message") -> Iterator[Tuple[str, Any]]:
    yield from _get_token_counts(message.usage)


@_stop_on_exception
def _get_llm_model_name_from_input(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    if model_name := arguments.get("model"):
        yield LLM_MODEL_NAME, model_name
        yield LLM_REQUEST_MODEL_NAME, model_name


@_stop_on_exception
def _get_llm_model_name_from_response(message: "Message") -> Iterator[Tuple[str, Any]]:
    if model_name := message.model:
        yield LLM_MODEL_NAME, model_name
        yield LLM_RESPONSE_MODEL_NAME, model_name


@_stop_on_exception
def _get_llm_input_messages(
    messages: Iterable[MessageParam],
    system: str | Iterable[TextBlockParam] | None = None,
    hide_images: bool = False,
) -> Iterator[Tuple[str, Any]]:
    """
    Extracts the messages from the chat response.

    The Anthropic Messages API takes `system` as a separate top-level parameter
    rather than as a system role entry in `messages`. When provided, prepend a
    synthetic system message so OI consumers see system content in
    LLM_INPUT_MESSAGES alongside user/assistant turns.

    The system message comes first, and each message is guarded individually, so a failure on
    one message keeps the indices of, and does not drop, the others.
    """
    if system:
        yield from _attributes_or_nothing(_get_llm_input_system_message, system, hide_images)
    i = 1 if system else 0
    for message in messages:
        content = _get(message, "content")
        blocks = (
            list(content) if not isinstance(content, str) and isinstance(content, Iterable) else []
        )
        if blocks:
            # Reuse the materialized content so one-shot SDK iterables are not consumed twice.
            message = message.copy()
            message["content"] = blocks
        tool_results = [block for block in blocks if _get(block, "type") == "tool_result"]
        # Each result needs its own message so parallel tool results cannot overwrite one another.
        # Anthropic requires tool results before the remaining content of a user message.
        for block in tool_results:
            yield f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_ROLE}", "tool"
            yield from _get_message_contents_attributes(
                f"{LLM_INPUT_MESSAGES}.{i}", [block], hide_images
            )
            i += 1
        if tool_results:
            remaining = [block for block in blocks if _get(block, "type") != "tool_result"]
            if not remaining:
                continue
            message["content"] = remaining
        yield from _attributes_or_nothing(_get_llm_input_message, i, message, hide_images)
        i += 1


def _get_llm_input_system_message(
    system: str | Iterable[TextBlockParam],
    hide_images: bool = False,
) -> Iterator[Tuple[str, Any]]:
    yield f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}", "system"
    if isinstance(system, str):
        yield f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}", system
        return
    yield from _get_message_contents_attributes(f"{LLM_INPUT_MESSAGES}.0", system, hide_images)


def _get_llm_input_message(
    i: int, message: MessageParam, hide_images: bool = False
) -> Iterator[Tuple[str, Any]]:
    if role := _require(message, "role"):
        yield f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_ROLE}", role
    if content := _require(message, "content"):
        if isinstance(content, str):
            yield f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_CONTENT}", content
            return
        yield from _get_message_contents_attributes(
            f"{LLM_INPUT_MESSAGES}.{i}", content, hide_images
        )


@_stop_on_exception
def _get_metadata(message: Message, span: _WithSpan) -> Iterator[Tuple[str, Any]]:
    yield from _get_metadata_attributes(
        message, span.context_metadata, span.config, span.request_messages
    )


@_stop_on_exception
def _get_output_messages(response: Message) -> Iterator[Tuple[str, Any]]:
    """
    Extracts the tool call information from the response
    """
    yield from _get_output_message_attributes(response)


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
LLM_REQUEST_MODEL_NAME = SpanAttributes.LLM_REQUEST_MODEL_NAME
LLM_RESPONSE_MODEL_NAME = SpanAttributes.LLM_RESPONSE_MODEL_NAME
LLM_FINISH_REASON = SpanAttributes.LLM_FINISH_REASON
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES
LLM_PROMPT_TEMPLATE_VERSION = SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION
LLM_TOOLS = SpanAttributes.LLM_TOOLS
IMAGE_URL = ImageAttributes.IMAGE_URL
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
MESSAGE_CONTENT_SIGNATURE = MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE
MESSAGE_CONTENT_DATA = MessageContentAttributes.MESSAGE_CONTENT_DATA
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
TOOL_NAME = ToolAttributes.TOOL_NAME
TOOL_DESCRIPTION = ToolAttributes.TOOL_DESCRIPTION
USER_ID = SpanAttributes.USER_ID
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
LLM_PROVIDER_ANTHROPIC = OpenInferenceLLMProviderValues.ANTHROPIC.value
LLM_SYSTEM_ANTHROPIC = OpenInferenceLLMSystemValues.ANTHROPIC.value
