"""Tests for Text-to-Speech instrumentation."""

from typing import Any, Generator, Iterator
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.elevenlabs import ElevenLabsInstrumentor
from openinference.semconv.trace import SpanAttributes


class MockTextToSpeechClient:
    """Mock TextToSpeechClient for testing."""

    def convert(
        self,
        voice_id: str,
        *,
        text: str,
        model_id: str = None,
        output_format: str = None,
        **kwargs: Any,
    ) -> Iterator[bytes]:
        """Mock convert that yields audio chunks."""
        yield b"audio_chunk_1"
        yield b"audio_chunk_2"
        yield b"audio_chunk_3"

    def convert_with_timestamps(
        self,
        voice_id: str,
        *,
        text: str,
        model_id: str = None,
        output_format: str = None,
        **kwargs: Any,
    ) -> Any:
        """Mock convert_with_timestamps that returns response object."""
        response = MagicMock()
        response.audio_base64 = "YXVkaW9fZGF0YQ=="  # "audio_data" in base64
        response.alignment = None
        return response

    def stream(
        self,
        voice_id: str,
        *,
        text: str,
        model_id: str = None,
        output_format: str = None,
        **kwargs: Any,
    ) -> Iterator[bytes]:
        """Mock stream that yields audio chunks."""
        yield b"stream_chunk_1"
        yield b"stream_chunk_2"

    def stream_with_timestamps(
        self,
        voice_id: str,
        *,
        text: str,
        model_id: str = None,
        output_format: str = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Mock stream_with_timestamps that yields response chunks."""
        chunk1 = MagicMock()
        chunk1.audio_base64 = "Y2h1bmsxYXVkaW8="  # "chunk1audio" in base64
        yield chunk1

        chunk2 = MagicMock()
        chunk2.audio_base64 = "Y2h1bmsyYXVkaW8="  # "chunk2audio" in base64
        yield chunk2


class MockAsyncTextToSpeechClient:
    """Mock AsyncTextToSpeechClient for testing."""

    async def convert(
        self,
        voice_id: str,
        *,
        text: str,
        model_id: str = None,
        output_format: str = None,
        **kwargs: Any,
    ) -> Any:
        """Mock async convert that returns async iterator."""

        async def _async_gen() -> Any:
            yield b"async_audio_chunk_1"
            yield b"async_audio_chunk_2"

        return _async_gen()

    async def convert_with_timestamps(
        self,
        voice_id: str,
        *,
        text: str,
        model_id: str = None,
        output_format: str = None,
        **kwargs: Any,
    ) -> Any:
        """Mock async convert_with_timestamps."""
        response = MagicMock()
        response.audio_base64 = "YXN5bmNfYXVkaW9fZGF0YQ=="
        return response

    async def stream(
        self,
        voice_id: str,
        *,
        text: str,
        model_id: str = None,
        output_format: str = None,
        **kwargs: Any,
    ) -> Any:
        """Mock async stream."""

        async def _async_gen() -> Any:
            yield b"async_stream_chunk_1"
            yield b"async_stream_chunk_2"

        return _async_gen()

    async def stream_with_timestamps(
        self,
        voice_id: str,
        *,
        text: str,
        model_id: str = None,
        output_format: str = None,
        **kwargs: Any,
    ) -> Any:
        """Mock async stream_with_timestamps."""

        async def _async_gen() -> Any:
            chunk = MagicMock()
            chunk.audio_base64 = "YXN5bmNfY2h1bms="
            yield chunk

        return _async_gen()


@pytest.fixture
def mock_elevenlabs() -> Generator[None, None, None]:
    """Mock the elevenlabs module."""
    # Create mock module structure
    mock_client_module = MagicMock()
    mock_client_module.TextToSpeechClient = MockTextToSpeechClient
    mock_client_module.AsyncTextToSpeechClient = MockAsyncTextToSpeechClient

    mock_conversation_module = MagicMock()
    mock_conversation_module.Conversation = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "elevenlabs": MagicMock(),
            "elevenlabs.text_to_speech": MagicMock(),
            "elevenlabs.text_to_speech.client": mock_client_module,
            "elevenlabs.conversational_ai": MagicMock(),
            "elevenlabs.conversational_ai.conversation": mock_conversation_module,
        },
    ):
        yield


class TestTextToSpeechInstrumentation:
    """Test TTS instrumentation."""

    def test_convert_creates_span(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_elevenlabs: None,
    ) -> None:
        """Test that convert() creates a span with correct attributes."""
        # Instrument with skip_dep_check to avoid dependency validation
        instrumentor = ElevenLabsInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            # Create client and make call
            from elevenlabs.text_to_speech.client import TextToSpeechClient

            client = TextToSpeechClient()
            audio_iter = client.convert(
                voice_id="test_voice_id",
                text="Hello, world!",
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )

            # Consume the iterator
            audio_data = list(audio_iter)

            # Verify audio data
            assert len(audio_data) == 3
            assert audio_data[0] == b"audio_chunk_1"

            # Verify span was created
            spans = in_memory_span_exporter.get_finished_spans()
            assert len(spans) == 1

            span = spans[0]
            assert span.name == "ElevenLabs.TextToSpeech"

            # Check attributes
            attrs = dict(span.attributes or {})
            assert attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "LLM"
            assert attrs.get(SpanAttributes.INPUT_VALUE) == "Hello, world!"
            assert attrs.get(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
            assert attrs.get(SpanAttributes.LLM_MODEL_NAME) == "eleven_multilingual_v2"
            assert attrs.get(SpanAttributes.LLM_PROVIDER) == "elevenlabs"
            assert attrs.get(SpanAttributes.LLM_SYSTEM) == "elevenlabs"
            assert attrs.get("elevenlabs.voice_id") == "test_voice_id"
            assert attrs.get("elevenlabs.output_format") == "mp3_44100_128"

        finally:
            instrumentor.uninstrument()

    def test_convert_with_timestamps_creates_span(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_elevenlabs: None,
    ) -> None:
        """Test that convert_with_timestamps() creates a span."""
        instrumentor = ElevenLabsInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            from elevenlabs.text_to_speech.client import TextToSpeechClient

            client = TextToSpeechClient()
            response = client.convert_with_timestamps(
                voice_id="test_voice",
                text="Hello with timestamps",
                model_id="eleven_multilingual_v2",
            )

            # Verify response
            assert response.audio_base64 is not None

            # Verify span
            spans = in_memory_span_exporter.get_finished_spans()
            assert len(spans) == 1

            span = spans[0]
            assert span.name == "ElevenLabs.TextToSpeech"

            attrs = dict(span.attributes or {})
            assert attrs.get(SpanAttributes.INPUT_VALUE) == "Hello with timestamps"

        finally:
            instrumentor.uninstrument()

    def test_stream_creates_span(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_elevenlabs: None,
    ) -> None:
        """Test that stream() creates a span."""
        instrumentor = ElevenLabsInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            from elevenlabs.text_to_speech.client import TextToSpeechClient

            client = TextToSpeechClient()
            stream = client.stream(
                voice_id="stream_voice",
                text="Streaming test",
                model_id="eleven_flash_v2",
            )

            # Consume stream
            chunks = list(stream)
            assert len(chunks) == 2

            # Verify span
            spans = in_memory_span_exporter.get_finished_spans()
            assert len(spans) == 1

            span = spans[0]
            attrs = dict(span.attributes or {})
            assert attrs.get(SpanAttributes.INPUT_VALUE) == "Streaming test"
            assert attrs.get("elevenlabs.voice_id") == "stream_voice"
            # Check that byte count was recorded
            assert "elevenlabs.audio_bytes" in attrs
            assert "elevenlabs.audio_chunks" in attrs

        finally:
            instrumentor.uninstrument()

    def test_error_handling(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """Test that errors are properly recorded on spans."""

        # Create an error-raising mock client class with all required methods
        class ErrorTextToSpeechClient:
            def convert(self, voice_id: str, *, text: str, **kwargs: Any) -> Iterator[bytes]:
                raise ValueError("Test error")

            def convert_with_timestamps(self, voice_id: str, *, text: str, **kwargs: Any) -> Any:
                pass

            def stream(self, voice_id: str, *, text: str, **kwargs: Any) -> Iterator[bytes]:
                yield b""

            def stream_with_timestamps(
                self, voice_id: str, *, text: str, **kwargs: Any
            ) -> Iterator[Any]:
                yield MagicMock()

        class ErrorAsyncTextToSpeechClient:
            async def convert(self, voice_id: str, *, text: str, **kwargs: Any) -> Any:
                pass

            async def convert_with_timestamps(
                self, voice_id: str, *, text: str, **kwargs: Any
            ) -> Any:
                pass

            async def stream(self, voice_id: str, *, text: str, **kwargs: Any) -> Any:
                pass

            async def stream_with_timestamps(
                self, voice_id: str, *, text: str, **kwargs: Any
            ) -> Any:
                pass

        # Create mock module structure with error-raising client
        mock_client_module = MagicMock()
        mock_client_module.TextToSpeechClient = ErrorTextToSpeechClient
        mock_client_module.AsyncTextToSpeechClient = ErrorAsyncTextToSpeechClient

        mock_conversation_module = MagicMock()
        mock_conversation_module.Conversation = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "elevenlabs": MagicMock(),
                "elevenlabs.text_to_speech": MagicMock(),
                "elevenlabs.text_to_speech.client": mock_client_module,
                "elevenlabs.conversational_ai": MagicMock(),
                "elevenlabs.conversational_ai.conversation": mock_conversation_module,
            },
        ):
            instrumentor = ElevenLabsInstrumentor()
            instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

            try:
                from elevenlabs.text_to_speech.client import TextToSpeechClient

                # Create a client that raises an error
                client = TextToSpeechClient()

                with pytest.raises(ValueError, match="Test error"):
                    client.convert(
                        voice_id="test",
                        text="Error test",
                    )

                # Verify span was created with error status
                spans = in_memory_span_exporter.get_finished_spans()
                assert len(spans) == 1

                span = spans[0]
                assert span.status.status_code.name == "ERROR"

            finally:
                instrumentor.uninstrument()


class TestTextToSpeechErrorHandling:
    """Test error handling in TTS wrappers."""

    def test_convert_with_timestamps_error_records_span(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """Test that convert_with_timestamps errors are properly recorded."""

        class ErrorTextToSpeechClient:
            def convert(self, voice_id: str, *, text: str, **kwargs: Any) -> Iterator[bytes]:
                yield b""

            def convert_with_timestamps(self, voice_id: str, *, text: str, **kwargs: Any) -> Any:
                raise RuntimeError("Timestamps API error")

            def stream(self, voice_id: str, *, text: str, **kwargs: Any) -> Iterator[bytes]:
                yield b""

            def stream_with_timestamps(
                self, voice_id: str, *, text: str, **kwargs: Any
            ) -> Iterator[Any]:
                yield MagicMock()

        class MockAsyncClient:
            async def convert(self, *args: Any, **kwargs: Any) -> Any:
                pass

            async def convert_with_timestamps(self, *args: Any, **kwargs: Any) -> Any:
                pass

            async def stream(self, *args: Any, **kwargs: Any) -> Any:
                pass

            async def stream_with_timestamps(self, *args: Any, **kwargs: Any) -> Any:
                pass

        mock_client_module = MagicMock()
        mock_client_module.TextToSpeechClient = ErrorTextToSpeechClient
        mock_client_module.AsyncTextToSpeechClient = MockAsyncClient

        mock_conversation_module = MagicMock()
        mock_conversation_module.Conversation = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "elevenlabs": MagicMock(),
                "elevenlabs.text_to_speech": MagicMock(),
                "elevenlabs.text_to_speech.client": mock_client_module,
                "elevenlabs.conversational_ai": MagicMock(),
                "elevenlabs.conversational_ai.conversation": mock_conversation_module,
            },
        ):
            instrumentor = ElevenLabsInstrumentor()
            instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

            try:
                from elevenlabs.text_to_speech.client import TextToSpeechClient

                client = TextToSpeechClient()

                with pytest.raises(RuntimeError, match="Timestamps API error"):
                    client.convert_with_timestamps(voice_id="test", text="Test")

                spans = in_memory_span_exporter.get_finished_spans()
                assert len(spans) == 1
                assert spans[0].status.status_code.name == "ERROR"
                assert "Timestamps API error" in spans[0].status.description

                # Verify exception was recorded
                events = spans[0].events
                assert len(events) == 1
                assert events[0].name == "exception"

            finally:
                instrumentor.uninstrument()

    def test_stream_error_records_span(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """Test that stream errors are properly recorded."""

        class ErrorTextToSpeechClient:
            def convert(self, voice_id: str, *, text: str, **kwargs: Any) -> Iterator[bytes]:
                yield b""

            def convert_with_timestamps(self, voice_id: str, *, text: str, **kwargs: Any) -> Any:
                return MagicMock()

            def stream(self, voice_id: str, *, text: str, **kwargs: Any) -> Iterator[bytes]:
                raise ConnectionError("Stream connection failed")

            def stream_with_timestamps(
                self, voice_id: str, *, text: str, **kwargs: Any
            ) -> Iterator[Any]:
                yield MagicMock()

        class MockAsyncClient:
            async def convert(self, *args: Any, **kwargs: Any) -> Any:
                pass

            async def convert_with_timestamps(self, *args: Any, **kwargs: Any) -> Any:
                pass

            async def stream(self, *args: Any, **kwargs: Any) -> Any:
                pass

            async def stream_with_timestamps(self, *args: Any, **kwargs: Any) -> Any:
                pass

        mock_client_module = MagicMock()
        mock_client_module.TextToSpeechClient = ErrorTextToSpeechClient
        mock_client_module.AsyncTextToSpeechClient = MockAsyncClient

        mock_conversation_module = MagicMock()
        mock_conversation_module.Conversation = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "elevenlabs": MagicMock(),
                "elevenlabs.text_to_speech": MagicMock(),
                "elevenlabs.text_to_speech.client": mock_client_module,
                "elevenlabs.conversational_ai": MagicMock(),
                "elevenlabs.conversational_ai.conversation": mock_conversation_module,
            },
        ):
            instrumentor = ElevenLabsInstrumentor()
            instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

            try:
                from elevenlabs.text_to_speech.client import TextToSpeechClient

                client = TextToSpeechClient()

                with pytest.raises(ConnectionError, match="Stream connection failed"):
                    client.stream(voice_id="test", text="Test")

                spans = in_memory_span_exporter.get_finished_spans()
                assert len(spans) == 1
                assert spans[0].status.status_code.name == "ERROR"

            finally:
                instrumentor.uninstrument()

    @pytest.mark.asyncio
    async def test_async_convert_error_records_span(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """Test that async convert errors are properly recorded."""

        class MockSyncClient:
            def convert(self, *args: Any, **kwargs: Any) -> Iterator[bytes]:
                yield b""

            def convert_with_timestamps(self, *args: Any, **kwargs: Any) -> Any:
                return MagicMock()

            def stream(self, *args: Any, **kwargs: Any) -> Iterator[bytes]:
                yield b""

            def stream_with_timestamps(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
                yield MagicMock()

        class ErrorAsyncTextToSpeechClient:
            def convert(self, voice_id: str, *, text: str, **kwargs: Any) -> Any:
                raise TimeoutError("Async API timeout")

            async def convert_with_timestamps(self, *args: Any, **kwargs: Any) -> Any:
                pass

            async def stream(self, *args: Any, **kwargs: Any) -> Any:
                pass

            async def stream_with_timestamps(self, *args: Any, **kwargs: Any) -> Any:
                pass

        mock_client_module = MagicMock()
        mock_client_module.TextToSpeechClient = MockSyncClient
        mock_client_module.AsyncTextToSpeechClient = ErrorAsyncTextToSpeechClient

        mock_conversation_module = MagicMock()
        mock_conversation_module.Conversation = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "elevenlabs": MagicMock(),
                "elevenlabs.text_to_speech": MagicMock(),
                "elevenlabs.text_to_speech.client": mock_client_module,
                "elevenlabs.conversational_ai": MagicMock(),
                "elevenlabs.conversational_ai.conversation": mock_conversation_module,
            },
        ):
            instrumentor = ElevenLabsInstrumentor()
            instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

            try:
                from elevenlabs.text_to_speech.client import AsyncTextToSpeechClient

                client = AsyncTextToSpeechClient()

                with pytest.raises(TimeoutError, match="Async API timeout"):
                    client.convert(voice_id="test", text="Test")

                spans = in_memory_span_exporter.get_finished_spans()
                assert len(spans) == 1
                assert spans[0].status.status_code.name == "ERROR"
                assert "Async API timeout" in spans[0].status.description

            finally:
                instrumentor.uninstrument()


class TestTextToSpeechAttributes:
    """Test attribute extraction."""

    def test_voice_settings_in_invocation_parameters(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_elevenlabs: None,
    ) -> None:
        """Test that voice_settings are captured in invocation parameters."""
        instrumentor = ElevenLabsInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            from elevenlabs.text_to_speech.client import TextToSpeechClient

            # Create mock voice settings
            voice_settings = {"stability": 0.5, "similarity_boost": 0.8}

            client = TextToSpeechClient()
            audio_iter = client.convert(
                voice_id="test_voice",
                text="Test",
                voice_settings=voice_settings,
            )
            list(audio_iter)

            spans = in_memory_span_exporter.get_finished_spans()
            span = spans[0]
            attrs = dict(span.attributes or {})

            # Check invocation parameters contain voice_settings
            inv_params = attrs.get(SpanAttributes.LLM_INVOCATION_PARAMETERS)
            assert inv_params is not None
            assert "voice_settings" in inv_params

        finally:
            instrumentor.uninstrument()
