"""Stream wrapper with a finish-once contract.

``stream=True`` chat calls return an iterator of ``ChatResponse`` chunks.
``_Stream`` proxies it (a ``wrapt.ObjectProxy``, so every attribute of the
underlying stream passes through) and finishes the span exactly once: on
exhaustion (OK), on any error including BaseExceptions like CancelledError
(ERROR, keeping the partial output that arrived), or on abandonment /
garbage-collection without iteration (status left UNSET to distinguish a
truncated stream from a completed one).
"""

import logging
from typing import Any, AsyncIterator, Iterator, List, Optional

from opentelemetry import trace as trace_api
from wrapt import ObjectProxy

from openinference.instrumentation.ollama._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.instrumentation.ollama._utils import _finish_tracing
from openinference.instrumentation.ollama._with_span import _WithSpan

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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


class _Stream(ObjectProxy):  # type: ignore[misc,type-arg,unused-ignore]
    """Wraps the (async)iterator returned by ``chat(stream=True)``."""

    __slots__ = (
        "_self_span",
        "_self_response_extractor",
        "_self_accumulator",
        "_self_finished",
    )

    def __init__(
        self,
        stream: Any,
        span: _WithSpan,
        response_extractor: _ResponseAttributesExtractor,
    ) -> None:
        super().__init__(stream)
        self._self_span = span
        self._self_response_extractor = response_extractor
        self._self_accumulator = _ChunkAccumulator()
        self._self_finished = False

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = self.__wrapped__.__next__()
        except StopIteration:
            self._finish(trace_api.Status(status_code=trace_api.StatusCode.OK))
            raise
        except BaseException as exception:
            self._finish_error(exception)
            raise
        self._self_accumulator.add(chunk)
        return chunk

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self.__wrapped__.__anext__()
        except StopAsyncIteration:
            self._finish(trace_api.Status(status_code=trace_api.StatusCode.OK))
            raise
        except BaseException as exception:
            self._finish_error(exception)
            raise
        self._self_accumulator.add(chunk)
        return chunk

    def close(self) -> None:
        try:
            close = getattr(self.__wrapped__, "close", None)
            if callable(close):
                close()
        finally:
            # Closed before exhaustion: leave the span status UNSET to
            # distinguish a truncated stream from a completed one.
            self._finish(None)

    async def aclose(self) -> None:
        try:
            aclose = getattr(self.__wrapped__, "aclose", None)
            if callable(aclose):
                await aclose()
        finally:
            self._finish(None)

    def __del__(self) -> None:
        # Abandoned (possibly never iterated): the span must still be ended.
        try:
            self._finish(None)
        except BaseException:
            pass

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

    def _finish(
        self,
        status: Optional[trace_api.Status],
        exception: Optional[BaseException] = None,
    ) -> None:
        if self._self_finished:
            return
        self._self_finished = True
        if exception is not None:
            self._self_span.record_exception(exception)
        response = self._self_accumulator.merged_response()
        if response is None:
            self._self_span.finish_tracing(status=status)
            return
        try:
            _finish_tracing(
                status=status,
                with_span=self._self_span,
                attributes=self._self_response_extractor.get_attributes(response=response),
                extra_attributes=self._self_response_extractor.get_extra_attributes(
                    response=response
                ),
            )
        except Exception:
            logger.exception(f"Failed to finalize streamed response of type {type(response)}")
            self._self_span.finish_tracing()
