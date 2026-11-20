from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Callable, ContextManager, Iterator, Mapping, Optional, cast

import wrapt
from botocore.eventstream import EventStream
from opentelemetry.trace import Span, Status, StatusCode, use_span
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.bedrock._proxy import _AnyT, _CallbackT, _Iterator
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _StreamClosed(StopIteration):
    """The stream was closed or dropped before it was read to the end."""


class _OnceCallback:
    """Passes events to the callback until the stream ends, then ignores further calls."""

    def __init__(self, callback: _CallbackT[Any]) -> None:
        self._callback = callback
        self.finished = False

    def __call__(self, obj: Any) -> Any:
        if self.finished:
            return obj
        if isinstance(obj, BaseException):
            self.finished = True
        return self._callback(obj)


class _AsyncIterator:
    """
    Async counterpart of _Iterator: wraps an async event stream, runs callback per
    item and on stream end/error, and manages a context (e.g. span) so it is
    closed when the stream is fully consumed or on exception.
    """

    def __init__(
        self,
        iterable: AsyncIterator[Any],
        callback: Optional[_CallbackT[_AnyT]] = None,
        context_manager_factory: Optional[Callable[[], ContextManager[Any]]] = None,
        stream: Any = None,
    ) -> None:
        self._iterator = iterable
        # Keeps the stream wrapper alive while this iterator is in use.
        self._stream = stream
        # Ensure we always have a callable callback.
        if callback is None:
            self._callback: Callable[[Any], Any] = lambda _value: None
        else:
            # cast to a generic Callable that accepts Any to avoid type incompatibilities
            self._callback = cast(Callable[[Any], Any], callback)
        self._context_manager_factory = context_manager_factory
        self._finished = False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        if self._finished:
            raise StopAsyncIteration
        cm = self._context_manager_factory() if self._context_manager_factory else None
        if cm is not None:
            cm.__enter__()
        try:
            value = await self._iterator.__anext__()
            if cm is not None:
                cm.__exit__(None, None, None)
            self._callback(value)
            return value
        except StopAsyncIteration as e:
            self._finished = True
            if cm is not None:
                cm.__exit__(None, None, None)
            self._callback(e)
            raise
        except Exception as e:
            if cm is not None:
                cm.__exit__(type(e), e, e.__traceback__)
            self._callback(e)
            raise


class _EventStreamIterator(_Iterator[Any]):
    def __init__(
        self,
        stream: _EventStream,
        callback: Optional[_CallbackT[_AnyT]] = None,
        context_manager_factory: Optional[Callable[[], ContextManager[Any]]] = None,
    ) -> None:
        super().__init__(iter(stream.__wrapped__), callback, context_manager_factory)
        # Keeps the stream wrapper alive while this iterator is in use.
        self._self_stream = stream

    def __iter__(self) -> _EventStreamIterator:
        # Returning a new iterator here would drop the reference to the stream.
        return self


class _EventStream(wrapt.ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    """
    Wraps a botocore event stream so the span ends when the stream is read to the end,
    fails, is closed, or is dropped.

    A stream that is closed or dropped early ends its span with the output read so far
    and leaves the status unset. Iterators created from the stream keep it alive, so a
    stream that is still being read is never treated as dropped.
    """

    __wrapped__: EventStream

    def __init__(
        self,
        obj: EventStream,
        callback: Optional[_CallbackT[_AnyT]] = None,
        context_manager_factory: Optional[Callable[[], ContextManager[Any]]] = None,
    ) -> None:
        super().__init__(obj)
        self._self_callback = _OnceCallback(callback) if callback is not None else None
        self._self_context_manager_factory = context_manager_factory
        # aiobotocore event streams provide __aiter__; botocore's are sync-only.
        self._self_is_async = hasattr(obj, "__aiter__")

    def __iter__(self) -> Iterator[Any]:
        return _EventStreamIterator(
            self,
            self._self_callback,
            self._self_context_manager_factory,
        )

    def __aiter__(self) -> AsyncIterator[Any]:
        if not self._self_is_async:
            raise TypeError("This is a sync stream. Use 'for' instead of 'async for'.")
        return _AsyncIterator(
            self.__wrapped__.__aiter__(),
            self._self_callback,
            self._self_context_manager_factory,
            stream=self,
        )

    def close(self) -> None:
        try:
            self.__wrapped__.close()
        finally:
            self._finish_early()

    def __del__(self) -> None:
        try:
            self._finish_early()
        except BaseException:
            pass

    def _finish_early(self) -> None:
        if (callback := self._self_callback) is None or callback.finished:
            return
        try:
            callback(_StreamClosed())
        except Exception:
            logger.exception("Failed to finish the span of a stream that was closed early")


def _use_span(span: Span) -> Callable[[], ContextManager[Span]]:
    # The `use_span` context manager can't be entered more than once. It would err here:
    # https://github.com/open-telemetry/opentelemetry-python/blob/b1e99c1555721f818e578d7457587693e767e182/opentelemetry-api/src/opentelemetry/util/_decorator.py#L56  # noqa E501
    # So we need a factory.
    return lambda: cast(ContextManager[Span], use_span(span, False, False, False))


def _finish(
    span: Span,
    result: Any,
    request_attributes: Mapping[str, AttributeValue],
    closed: bool = False,
) -> None:
    if isinstance(result, BaseException):
        span.record_exception(result)
        span.set_status(Status(StatusCode.ERROR, f"{type(result).__name__}: {result}"))
        for k, v in request_attributes.items():
            span.set_attribute(k, v)
        span.end()
        return
    if isinstance(result, dict):
        span.set_attribute(OUTPUT_VALUE, safe_json_dumps(result))
        span.set_attribute(OUTPUT_MIME_TYPE, JSON)
    elif result is not None:
        span.set_attribute(OUTPUT_VALUE, str(result))
    # A stream closed before its end keeps the status unset: the output may be incomplete.
    if not closed:
        span.set_status(Status(StatusCode.OK))
    for k, v in request_attributes.items():
        span.set_attribute(k, v)
    span.end()


IMAGE_URL = ImageAttributes.IMAGE_URL
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
JSON = OpenInferenceMimeTypeValues.JSON.value
LLM = OpenInferenceSpanKindValues.LLM.value
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_NAME = MessageAttributes.MESSAGE_NAME
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
