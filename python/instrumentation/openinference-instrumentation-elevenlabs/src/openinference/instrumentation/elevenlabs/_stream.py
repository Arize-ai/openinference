"""Audio stream wrappers for ElevenLabs instrumentation."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Iterator, Optional

from opentelemetry import trace as trace_api
from wrapt import ObjectProxy

from ._attributes import get_tts_response_attributes
from ._with_span import _WithSpan

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _AudioStream(ObjectProxy):  # type: ignore[misc]
    """
    Wrapper for synchronous audio byte iterators.

    Tracks the stream lifecycle and finalizes the span when the stream
    is exhausted or an error occurs.
    """

    __slots__ = (
        "_self_with_span",
        "_self_byte_count",
        "_self_chunk_count",
        "_self_text",
        "_self_output_format",
        "_self_is_finished",
    )

    def __init__(
        self,
        stream: Iterator[bytes],
        with_span: _WithSpan,
        text: str,
        output_format: Optional[str],
    ) -> None:
        super().__init__(stream)
        self._self_with_span = with_span
        self._self_byte_count = 0
        self._self_chunk_count = 0
        self._self_text = text
        self._self_output_format = output_format
        self._self_is_finished = False

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        if self._self_is_finished:
            raise StopIteration

        try:
            chunk = self.__wrapped__.__next__()
            self._self_byte_count += len(chunk)
            self._self_chunk_count += 1
            return chunk
        except StopIteration:
            self._finish_tracing(status=trace_api.Status(trace_api.StatusCode.OK))
            raise
        except Exception as exception:
            self._finish_tracing(
                status=trace_api.Status(trace_api.StatusCode.ERROR, str(exception)),
                exception=exception,
            )
            raise

    def _finish_tracing(
        self,
        status: trace_api.Status,
        exception: Optional[Exception] = None,
    ) -> None:
        """Finish tracing with final attributes."""
        if self._self_is_finished:
            return

        self._self_is_finished = True

        if exception:
            self._self_with_span.record_exception(exception)

        # Build response attributes
        response_attrs = dict(
            get_tts_response_attributes(
                text=self._self_text,
                output_format=self._self_output_format,
                byte_count=self._self_byte_count,
                chunk_count=self._self_chunk_count,
            )
        )

        self._self_with_span.finish_tracing(
            status=status,
            attributes=response_attrs,
        )


class _AsyncAudioStream(ObjectProxy):  # type: ignore[misc]
    """
    Wrapper for asynchronous audio byte iterators.

    Tracks the stream lifecycle and finalizes the span when the stream
    is exhausted or an error occurs.
    """

    __slots__ = (
        "_self_with_span",
        "_self_byte_count",
        "_self_chunk_count",
        "_self_text",
        "_self_output_format",
        "_self_is_finished",
    )

    def __init__(
        self,
        stream: AsyncIterator[bytes],
        with_span: _WithSpan,
        text: str,
        output_format: Optional[str],
    ) -> None:
        super().__init__(stream)
        self._self_with_span = with_span
        self._self_byte_count = 0
        self._self_chunk_count = 0
        self._self_text = text
        self._self_output_format = output_format
        self._self_is_finished = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self

    async def __anext__(self) -> bytes:
        if self._self_is_finished:
            raise StopAsyncIteration

        try:
            chunk = await self.__wrapped__.__anext__()
            self._self_byte_count += len(chunk)
            self._self_chunk_count += 1
            return chunk
        except StopAsyncIteration:
            self._finish_tracing(status=trace_api.Status(trace_api.StatusCode.OK))
            raise
        except Exception as exception:
            self._finish_tracing(
                status=trace_api.Status(trace_api.StatusCode.ERROR, str(exception)),
                exception=exception,
            )
            raise

    def _finish_tracing(
        self,
        status: trace_api.Status,
        exception: Optional[Exception] = None,
    ) -> None:
        """Finish tracing with final attributes."""
        if self._self_is_finished:
            return

        self._self_is_finished = True

        if exception:
            self._self_with_span.record_exception(exception)

        # Build response attributes
        response_attrs = dict(
            get_tts_response_attributes(
                text=self._self_text,
                output_format=self._self_output_format,
                byte_count=self._self_byte_count,
                chunk_count=self._self_chunk_count,
            )
        )

        self._self_with_span.finish_tracing(
            status=status,
            attributes=response_attrs,
        )


class _AsyncTimestampedAudioStream(ObjectProxy):  # type: ignore[misc]
    """
    Wrapper for stream_with_timestamps iterators.

    These streams return StreamingAudioChunkWithTimestampsResponse objects
    instead of raw bytes.
    """

    __slots__ = (
        "_self_with_span",
        "_self_byte_count",
        "_self_chunk_count",
        "_self_text",
        "_self_output_format",
        "_self_is_finished",
        "_self_is_async",
    )

    def __init__(
        self,
        stream: Any,  # Iterator or AsyncIterator of StreamingAudioChunkWithTimestampsResponse
        with_span: _WithSpan,
        text: str,
        output_format: Optional[str],
        is_async: bool = False,
    ) -> None:
        super().__init__(stream)
        self._self_with_span = with_span
        self._self_byte_count = 0
        self._self_chunk_count = 0
        self._self_text = text
        self._self_output_format = output_format
        self._self_is_finished = False
        self._self_is_async = is_async

    def __iter__(self) -> Iterator[Any]:
        return self

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    def __next__(self) -> Any:
        if self._self_is_finished:
            raise StopIteration

        try:
            chunk = self.__wrapped__.__next__()
            self._process_chunk(chunk)
            return chunk
        except StopIteration:
            self._finish_tracing(status=trace_api.Status(trace_api.StatusCode.OK))
            raise
        except Exception as exception:
            self._finish_tracing(
                status=trace_api.Status(trace_api.StatusCode.ERROR, str(exception)),
                exception=exception,
            )
            raise

    async def __anext__(self) -> Any:
        if self._self_is_finished:
            raise StopAsyncIteration

        try:
            chunk = await self.__wrapped__.__anext__()
            self._process_chunk(chunk)
            return chunk
        except StopAsyncIteration:
            self._finish_tracing(status=trace_api.Status(trace_api.StatusCode.OK))
            raise
        except Exception as exception:
            self._finish_tracing(
                status=trace_api.Status(trace_api.StatusCode.ERROR, str(exception)),
                exception=exception,
            )
            raise

    def _process_chunk(self, chunk: Any) -> None:
        """Process a chunk to extract byte count."""
        self._self_chunk_count += 1

        # StreamingAudioChunkWithTimestampsResponse has audio_base64 attribute
        if hasattr(chunk, "audio_base64") and chunk.audio_base64:
            # Base64 encoded, actual bytes are ~75% of string length
            self._self_byte_count += int(len(chunk.audio_base64) * 0.75)
        elif hasattr(chunk, "audio") and chunk.audio:
            # Some responses may have raw audio bytes
            if isinstance(chunk.audio, bytes):
                self._self_byte_count += len(chunk.audio)

    def _finish_tracing(
        self,
        status: trace_api.Status,
        exception: Optional[Exception] = None,
    ) -> None:
        """Finish tracing with final attributes."""
        if self._self_is_finished:
            return

        self._self_is_finished = True

        if exception:
            self._self_with_span.record_exception(exception)

        # Build response attributes
        response_attrs = dict(
            get_tts_response_attributes(
                text=self._self_text,
                output_format=self._self_output_format,
                byte_count=self._self_byte_count,
                chunk_count=self._self_chunk_count,
            )
        )

        self._self_with_span.finish_tracing(
            status=status,
            attributes=response_attrs,
        )
