import logging
from abc import ABC
from collections.abc import AsyncIterator as AsyncIteratorABC
from collections.abc import Iterator as IteratorABC
from contextlib import contextmanager
from inspect import Signature, signature
from typing import (
    Any,
    AsyncIterator,
    Callable,
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
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import get_attributes_from_context
from openinference.instrumentation.ollama._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.instrumentation.ollama._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.instrumentation.ollama._utils import _finish_tracing
from openinference.instrumentation.ollama._with_span import _WithSpan

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _WithTracer(ABC):
    """Base class for wrappers that need a tracer."""

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer

    @contextmanager
    def _start_as_current_span(
        self,
        span_name: str,
        attributes: Iterable[Tuple[str, AttributeValue]],
        context_attributes: Iterable[Tuple[str, AttributeValue]],
        extra_attributes: Iterable[Tuple[str, AttributeValue]],
    ) -> Iterator[_WithSpan]:
        # Because OTEL has a default limit of 128 attributes, we split our
        # attributes into two tiers, where "extra_attributes" are added first to
        # ensure that the most important "attributes" are added last and are not
        # dropped.
        try:
            span = self._tracer.start_span(name=span_name, attributes=dict(extra_attributes))
        except Exception:
            span = INVALID_SPAN
        with trace_api.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            yield _WithSpan(
                span=span,
                context_attributes=dict(context_attributes),
                extra_attributes=dict(attributes),
            )


def _parse_args(
    signature: Signature,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    bound_signature = signature.bind(*args, **kwargs)
    bound_signature.apply_defaults()
    return {
        key: value
        for key, value in bound_signature.arguments.items()
        if value is not None and key != "self"
    }


class _ChunkAccumulator:
    """Accumulates streamed ``ChatResponse`` chunks without retaining them all:
    only the concatenated content/thinking fragments, the tool calls, and the
    final chunk (which carries the model and token counts) are kept."""

    def __init__(self) -> None:
        self._content: List[str] = []
        self._thinking: List[str] = []
        self._tool_calls: List[Any] = []
        self._last_chunk: Any = None

    def add(self, chunk: Any) -> None:
        self._last_chunk = chunk
        message = getattr(chunk, "message", None)
        if message is None:
            return
        if content := getattr(message, "content", None):
            self._content.append(content)
        if thinking := getattr(message, "thinking", None):
            self._thinking.append(thinking)
        if tool_calls := getattr(message, "tool_calls", None):
            self._tool_calls.extend(tool_calls)

    def merged_response(self) -> Any:
        """A single response-like object combining all received chunks, or
        None if no chunks arrived."""
        if self._last_chunk is None:
            return None
        try:
            merged = self._last_chunk.model_copy(deep=True)
            merged.message.content = "".join(self._content)
            if self._thinking:
                merged.message.thinking = "".join(self._thinking)
            if self._tool_calls:
                merged.message.tool_calls = list(self._tool_calls)
            return merged
        except Exception:
            logger.exception("Failed to merge streamed chat chunks")
            return self._last_chunk


class _ChatWrapperBase(_WithTracer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_extractor = _RequestAttributesExtractor()
        self._response_extractor = _ResponseAttributesExtractor()
        self._wrapped_signature: Optional[Signature] = None

    def _parse_request(
        self, wrapped: Callable[..., Any], args: Tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> Dict[str, Any]:
        if self._wrapped_signature is None:
            self._wrapped_signature = signature(wrapped)
        return _parse_args(self._wrapped_signature, *args, **kwargs)

    def _record_exception(self, span: _WithSpan, exception: BaseException) -> None:
        span.record_exception(exception)
        span.finish_tracing(
            status=trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
        )

    def _finalize_response(
        self,
        span: _WithSpan,
        response: Any,
        status: Optional[trace_api.Status],
    ) -> None:
        """Finishes the span with attributes extracted from the response. A
        None status leaves the span status UNSET (used for streams that were
        abandoned before completing)."""
        if response is None:
            span.finish_tracing(status=status)
            return
        try:
            _finish_tracing(
                status=status,
                with_span=span,
                attributes=self._response_extractor.get_attributes(response=response),
                extra_attributes=self._response_extractor.get_extra_attributes(response=response),
            )
        except Exception:
            logger.exception(f"Failed to finalize response of type {type(response)}")
            span.finish_tracing()


class _StreamBase:
    """Shared bookkeeping for stream wrappers: accumulates chunks and
    guarantees the span is finished exactly once — on exhaustion, on error
    (including BaseExceptions like CancelledError), or when the wrapper is
    garbage-collected without ever being consumed."""

    def __init__(self, span: _WithSpan, wrapper: _ChatWrapperBase) -> None:
        self._span = span
        self._wrapper = wrapper
        self._accumulator = _ChunkAccumulator()
        self._finished = False

    def _finish(
        self,
        status: Optional[trace_api.Status],
        exception: Optional[BaseException] = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        if exception is not None:
            self._span.record_exception(exception)
        self._wrapper._finalize_response(
            self._span, self._accumulator.merged_response(), status=status
        )

    def _finish_error(self, exception: BaseException) -> None:
        # Record the exception but keep the partial output that arrived
        # before the failure.
        self._finish(
            trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            ),
            exception=exception,
        )

    def __del__(self) -> None:
        # Abandoned before completing: leave the span status UNSET to
        # distinguish a truncated stream from a completed one.
        try:
            self._finish(None)
        except BaseException:
            pass


class _Stream(_StreamBase):
    def __init__(self, stream: Iterator[Any], span: _WithSpan, wrapper: _ChatWrapperBase) -> None:
        super().__init__(span=span, wrapper=wrapper)
        self._stream = stream

    def __iter__(self) -> "_Stream":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._finish(trace_api.Status(status_code=trace_api.StatusCode.OK))
            raise
        except BaseException as exception:
            self._finish_error(exception)
            raise
        self._accumulator.add(chunk)
        return chunk

    def close(self) -> None:
        try:
            close = getattr(self._stream, "close", None)
            if callable(close):
                close()
        finally:
            self._finish(None)


class _AsyncStream(_StreamBase):
    def __init__(
        self, stream: AsyncIterator[Any], span: _WithSpan, wrapper: _ChatWrapperBase
    ) -> None:
        super().__init__(span=span, wrapper=wrapper)
        self._stream = stream

    def __aiter__(self) -> "_AsyncStream":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            self._finish(trace_api.Status(status_code=trace_api.StatusCode.OK))
            raise
        except BaseException as exception:
            self._finish_error(exception)
            raise
        self._accumulator.add(chunk)
        return chunk

    async def aclose(self) -> None:
        try:
            aclose = getattr(self._stream, "aclose", None)
            if callable(aclose):
                await aclose()
        finally:
            self._finish(None)


class _ChatWrapper(_ChatWrapperBase):
    """Wraps ``ollama.Client.chat`` to trace synchronous chat calls."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        request_parameters = self._parse_request(wrapped, args, kwargs)
        with self._start_as_current_span(
            span_name="chat",
            attributes=self._request_extractor.get_attributes_from_request(request_parameters),
            context_attributes=get_attributes_from_context(),
            extra_attributes=self._request_extractor.get_extra_attributes_from_request(
                request_parameters
            ),
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                self._record_exception(span, exception)
                raise
            if isinstance(response, IteratorABC):
                # ``stream=True``: the span is finished when the stream is
                # exhausted, fails, or is abandoned — not when the call returns.
                return _Stream(response, span, self)
            self._finalize_response(
                span, response, status=trace_api.Status(status_code=trace_api.StatusCode.OK)
            )
        return response


class _AsyncChatWrapper(_ChatWrapperBase):
    """Wraps ``ollama.AsyncClient.chat`` to trace asynchronous chat calls."""

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        request_parameters = self._parse_request(wrapped, args, kwargs)
        with self._start_as_current_span(
            span_name="async_chat",
            attributes=self._request_extractor.get_attributes_from_request(request_parameters),
            context_attributes=get_attributes_from_context(),
            extra_attributes=self._request_extractor.get_extra_attributes_from_request(
                request_parameters
            ),
        ) as span:
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                self._record_exception(span, exception)
                raise
            if isinstance(response, AsyncIteratorABC):
                # ``stream=True``: the span is finished when the stream is
                # exhausted, fails, or is abandoned — not when the call returns.
                return _AsyncStream(response, span, self)
            self._finalize_response(
                span, response, status=trace_api.Status(status_code=trace_api.StatusCode.OK)
            )
        return response
