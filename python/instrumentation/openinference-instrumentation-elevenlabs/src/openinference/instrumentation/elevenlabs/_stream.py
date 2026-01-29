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
            chunk: bytes = self.__wrapped__.__next__()
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


class _AsyncAudioStream:
    """
    Wrapper for asynchronous audio byte iterators.

    Tracks the stream lifecycle and finalizes the span when the stream
    is exhausted or an error occurs.
    """

    __slots__ = (
        "_stream",
        "_with_span",
        "_byte_count",
        "_chunk_count",
        "_text",
        "_output_format",
        "_is_finished",
    )

    def __init__(
        self,
        stream: AsyncIterator[bytes],
        with_span: _WithSpan,
        text: str,
        output_format: Optional[str],
    ) -> None:
        self._stream = stream
        self._with_span = with_span
        self._byte_count = 0
        self._chunk_count = 0
        self._text = text
        self._output_format = output_format
        self._is_finished = False

    def __aiter__(self) -> "_AsyncAudioStream":
        return self

    async def __anext__(self) -> bytes:
        if self._is_finished:
            raise StopAsyncIteration

        try:
            chunk: bytes = await self._stream.__anext__()
            self._byte_count += len(chunk)
            self._chunk_count += 1
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
        if self._is_finished:
            return

        self._is_finished = True

        if exception:
            self._with_span.record_exception(exception)

        # Build response attributes
        response_attrs = dict(
            get_tts_response_attributes(
                text=self._text,
                output_format=self._output_format,
                byte_count=self._byte_count,
                chunk_count=self._chunk_count,
            )
        )

        self._with_span.finish_tracing(
            status=status,
            attributes=response_attrs,
        )


class _TimestampedAudioStream:
    """
    Wrapper for sync convert_with_timestamps/stream_with_timestamps iterators.

    These streams return objects with audio_base_64 attribute instead of raw bytes.
    """

    __slots__ = (
        "_stream",
        "_with_span",
        "_byte_count",
        "_chunk_count",
        "_text",
        "_output_format",
        "_is_finished",
    )

    def __init__(
        self,
        stream: Iterator[Any],
        with_span: _WithSpan,
        text: str,
        output_format: Optional[str],
    ) -> None:
        self._stream = stream
        self._with_span = with_span
        self._byte_count = 0
        self._chunk_count = 0
        self._text = text
        self._output_format = output_format
        self._is_finished = False

    def __iter__(self) -> "_TimestampedAudioStream":
        return self

    def __next__(self) -> Any:
        if self._is_finished:
            raise StopIteration

        try:
            chunk = self._stream.__next__()
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

    def _process_chunk(self, chunk: Any) -> None:
        """Process a chunk to extract byte count."""
        self._chunk_count += 1

        # Response chunks have audio_base_64 attribute
        if hasattr(chunk, "audio_base_64") and chunk.audio_base_64:
            # Base64 encoded, actual bytes are ~75% of string length
            self._byte_count += int(len(chunk.audio_base_64) * 0.75)
        elif hasattr(chunk, "audio") and chunk.audio:
            # Some responses may have raw audio bytes
            if isinstance(chunk.audio, bytes):
                self._byte_count += len(chunk.audio)

    def _finish_tracing(
        self,
        status: trace_api.Status,
        exception: Optional[Exception] = None,
    ) -> None:
        """Finish tracing with final attributes."""
        if self._is_finished:
            return

        self._is_finished = True

        if exception:
            self._with_span.record_exception(exception)

        # Build response attributes
        response_attrs = dict(
            get_tts_response_attributes(
                text=self._text,
                output_format=self._output_format,
                byte_count=self._byte_count,
                chunk_count=self._chunk_count,
            )
        )

        self._with_span.finish_tracing(
            status=status,
            attributes=response_attrs,
        )


class _AsyncTimestampedAudioStream:
    """
    Wrapper for async convert_with_timestamps/stream_with_timestamps iterators.

    These streams return objects with audio_base_64 attribute instead of raw bytes.
    """

    __slots__ = (
        "_stream",
        "_with_span",
        "_byte_count",
        "_chunk_count",
        "_text",
        "_output_format",
        "_is_finished",
    )

    def __init__(
        self,
        stream: AsyncIterator[Any],
        with_span: _WithSpan,
        text: str,
        output_format: Optional[str],
    ) -> None:
        self._stream = stream
        self._with_span = with_span
        self._byte_count = 0
        self._chunk_count = 0
        self._text = text
        self._output_format = output_format
        self._is_finished = False

    def __aiter__(self) -> "_AsyncTimestampedAudioStream":
        return self

    async def __anext__(self) -> Any:
        if self._is_finished:
            raise StopAsyncIteration

        try:
            chunk = await self._stream.__anext__()
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
        self._chunk_count += 1

        # Response chunks have audio_base_64 attribute
        if hasattr(chunk, "audio_base_64") and chunk.audio_base_64:
            # Base64 encoded, actual bytes are ~75% of string length
            self._byte_count += int(len(chunk.audio_base_64) * 0.75)
        elif hasattr(chunk, "audio") and chunk.audio:
            # Some responses may have raw audio bytes
            if isinstance(chunk.audio, bytes):
                self._byte_count += len(chunk.audio)

    def _finish_tracing(
        self,
        status: trace_api.Status,
        exception: Optional[Exception] = None,
    ) -> None:
        """Finish tracing with final attributes."""
        if self._is_finished:
            return

        self._is_finished = True

        if exception:
            self._with_span.record_exception(exception)

        # Build response attributes
        response_attrs = dict(
            get_tts_response_attributes(
                text=self._text,
                output_format=self._output_format,
                byte_count=self._byte_count,
                chunk_count=self._chunk_count,
            )
        )

        self._with_span.finish_tracing(
            status=status,
            attributes=response_attrs,
        )
