"""Tests for audio streaming instrumentation."""

from typing import Any, AsyncIterator, Iterator
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.elevenlabs._attributes import (
    ELEVENLABS_AUDIO_BYTES,
    ELEVENLABS_AUDIO_CHUNKS,
)
from openinference.instrumentation.elevenlabs._stream import (
    _AsyncAudioStream,
    _AudioStream,
    _TimestampedAudioStream,
)
from openinference.instrumentation.elevenlabs._with_span import _WithSpan


class TestAudioStream:
    """Test synchronous audio stream wrapper."""

    def test_audio_stream_tracks_bytes_and_chunks(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """Test that AudioStream tracks byte count and chunk count."""
        # Create a span
        tracer = tracer_provider.get_tracer(__name__)
        span = tracer.start_span("test_stream")
        with_span = _WithSpan(span)

        # Create mock iterator
        def mock_iterator() -> Iterator[bytes]:
            yield b"chunk1"  # 6 bytes
            yield b"chunk2"  # 6 bytes
            yield b"chunk3"  # 6 bytes

        # Wrap the iterator
        stream = _AudioStream(
            stream=mock_iterator(),
            with_span=with_span,
            text="test text",
            output_format="mp3_44100_128",
        )

        # Consume the stream
        chunks = list(stream)

        # Verify chunks were returned
        assert len(chunks) == 3
        assert chunks[0] == b"chunk1"

        # Verify span finished with correct attributes
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1

        attrs = dict(spans[0].attributes or {})
        assert attrs.get(ELEVENLABS_AUDIO_BYTES) == 18  # 6 * 3
        assert attrs.get(ELEVENLABS_AUDIO_CHUNKS) == 3

    def test_audio_stream_handles_error(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """Test that AudioStream handles errors and records them."""
        tracer = tracer_provider.get_tracer(__name__)
        span = tracer.start_span("test_stream_error")
        with_span = _WithSpan(span)

        # Create iterator that raises error
        def error_iterator() -> Iterator[bytes]:
            yield b"chunk1"
            raise ValueError("Stream error")

        stream = _AudioStream(
            stream=error_iterator(),
            with_span=with_span,
            text="test",
            output_format=None,
        )

        # Consume until error
        with pytest.raises(ValueError, match="Stream error"):
            list(stream)

        # Verify span was finished with error status
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == trace_api.StatusCode.ERROR

    def test_audio_stream_empty_iterator(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """Test AudioStream with empty iterator."""
        tracer = tracer_provider.get_tracer(__name__)
        span = tracer.start_span("test_empty")
        with_span = _WithSpan(span)

        def empty_iterator() -> Iterator[bytes]:
            return
            yield  # Makes it a generator

        stream = _AudioStream(
            stream=empty_iterator(),
            with_span=with_span,
            text="",
            output_format=None,
        )

        chunks = list(stream)
        assert len(chunks) == 0

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        # For empty iterator, these might be 0 or not set
        audio_bytes = attrs.get(ELEVENLABS_AUDIO_BYTES, 0)
        audio_chunks = attrs.get(ELEVENLABS_AUDIO_CHUNKS, 0)
        assert audio_bytes == 0
        assert audio_chunks == 0


class TestAsyncAudioStream:
    """Test asynchronous audio stream wrapper."""

    @pytest.mark.asyncio
    async def test_async_audio_stream_tracks_bytes(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """Test that AsyncAudioStream tracks bytes and chunks."""
        tracer = tracer_provider.get_tracer(__name__)
        span = tracer.start_span("test_async_stream")
        with_span = _WithSpan(span)

        async def mock_async_iterator() -> AsyncIterator[bytes]:
            yield b"async1"  # 6 bytes
            yield b"async2"  # 6 bytes

        stream = _AsyncAudioStream(
            stream=mock_async_iterator(),
            with_span=with_span,
            text="async test",
            output_format="pcm_44100",
        )

        # Consume async stream
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0] == b"async1"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1

        attrs = dict(spans[0].attributes or {})
        assert attrs.get(ELEVENLABS_AUDIO_BYTES) == 12
        assert attrs.get(ELEVENLABS_AUDIO_CHUNKS) == 2

    @pytest.mark.asyncio
    async def test_async_audio_stream_handles_error(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """Test that AsyncAudioStream handles errors."""
        tracer = tracer_provider.get_tracer(__name__)
        span = tracer.start_span("test_async_error")
        with_span = _WithSpan(span)

        async def error_async_iterator() -> AsyncIterator[bytes]:
            yield b"chunk"
            raise RuntimeError("Async error")

        stream = _AsyncAudioStream(
            stream=error_async_iterator(),
            with_span=with_span,
            text="test",
            output_format=None,
        )

        with pytest.raises(RuntimeError, match="Async error"):
            async for _ in stream:
                pass

        spans = in_memory_span_exporter.get_finished_spans()
        assert spans[0].status.status_code == trace_api.StatusCode.ERROR


class TestTimestampedAudioStream:
    """Test timestamped audio stream wrapper."""

    def test_timestamped_stream_extracts_bytes_from_base64(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """Test that timestamped stream extracts byte count from base64."""
        tracer = tracer_provider.get_tracer(__name__)
        span = tracer.start_span("test_timestamps")
        with_span = _WithSpan(span)

        def mock_timestamped_iterator() -> Iterator[Any]:
            chunk1 = MagicMock()
            chunk1.audio_base_64 = "YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXo="  # 26 chars = ~19 bytes
            yield chunk1

            chunk2 = MagicMock()
            chunk2.audio_base_64 = "MTIzNDU2Nzg5MA=="  # 16 chars = ~12 bytes
            yield chunk2

        stream = _TimestampedAudioStream(
            stream=mock_timestamped_iterator(),
            with_span=with_span,
            text="timestamps test",
            output_format="mp3_44100_128",
        )

        chunks = list(stream)
        assert len(chunks) == 2

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1

        attrs = dict(spans[0].attributes or {})
        # Verify bytes were extracted (base64 length * 0.75)
        assert ELEVENLABS_AUDIO_BYTES in attrs
        assert attrs.get(ELEVENLABS_AUDIO_CHUNKS) == 2


class TestWithSpan:
    """Test _WithSpan helper class."""

    def test_with_span_finish_tracing_once(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """Test that finish_tracing can only be called once."""
        tracer = tracer_provider.get_tracer(__name__)
        span = tracer.start_span("test_finish")
        with_span = _WithSpan(span)

        # First finish should work
        with_span.finish_tracing(
            status=trace_api.Status(trace_api.StatusCode.OK),
            attributes={"key1": "value1"},
        )

        assert with_span.is_finished

        # Second finish should be no-op
        with_span.finish_tracing(
            status=trace_api.Status(trace_api.StatusCode.ERROR),
            attributes={"key2": "value2"},
        )

        # Only one span should exist with first attributes
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == trace_api.StatusCode.OK

    def test_with_span_record_exception(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """Test recording exceptions on span."""
        tracer = tracer_provider.get_tracer(__name__)
        span = tracer.start_span("test_exception")
        with_span = _WithSpan(span)

        exception = ValueError("Test exception")
        with_span.record_exception(exception)
        with_span.finish_tracing(
            status=trace_api.Status(trace_api.StatusCode.ERROR, str(exception))
        )

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1

        # Verify exception was recorded
        events = spans[0].events
        assert len(events) == 1
        assert events[0].name == "exception"

    def test_with_span_set_attributes(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """Test setting attributes on span."""
        tracer = tracer_provider.get_tracer(__name__)
        span = tracer.start_span("test_attrs")
        with_span = _WithSpan(span)

        with_span.set_attributes({"attr1": "value1", "attr2": 42})
        with_span.finish_tracing()

        spans = in_memory_span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes or {})
        assert attrs.get("attr1") == "value1"
        assert attrs.get("attr2") == 42
