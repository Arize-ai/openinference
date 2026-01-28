"""Tests for Conversation instrumentation."""

from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.elevenlabs import ElevenLabsInstrumentor
from openinference.instrumentation.elevenlabs._attributes import (
    ELEVENLABS_AGENT_ID,
    ELEVENLABS_CONVERSATION_ID,
)
from openinference.semconv.trace import SpanAttributes


class MockConversation:
    """Mock Conversation class for testing."""

    def __init__(
        self,
        client: Any = None,
        agent_id: str = None,
        user_id: str = None,
        requires_auth: bool = False,
        audio_interface: Any = None,
        **kwargs: Any,
    ) -> None:
        self.client = client
        self.agent_id = agent_id
        self.user_id = user_id
        self.requires_auth = requires_auth
        self._session_started = False
        self._session_ended = False
        self._conversation_id = "test_conversation_123"

    def start_session(self) -> None:
        """Start the conversation session."""
        self._session_started = True

    def end_session(self) -> None:
        """End the conversation session."""
        self._session_ended = True

    def wait_for_session_end(self) -> str:
        """Wait for session to end and return conversation ID."""
        self._session_ended = True
        return self._conversation_id


@pytest.fixture
def mock_conversation_module() -> Generator[None, None, None]:
    """Mock the elevenlabs conversation module."""
    mock_module = MagicMock()
    mock_module.Conversation = MockConversation

    # Also mock TTS to prevent import errors
    mock_tts_module = MagicMock()
    mock_tts_module.TextToSpeechClient = MagicMock()
    mock_tts_module.AsyncTextToSpeechClient = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "elevenlabs": MagicMock(),
            "elevenlabs.text_to_speech": MagicMock(),
            "elevenlabs.text_to_speech.client": mock_tts_module,
            "elevenlabs.conversational_ai": MagicMock(),
            "elevenlabs.conversational_ai.conversation": mock_module,
        },
    ):
        yield


class TestConversationInstrumentation:
    """Test Conversation class instrumentation."""

    def test_conversation_start_session_creates_span(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_conversation_module: None,
    ) -> None:
        """Test that start_session() creates a CHAIN span."""
        instrumentor = ElevenLabsInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            from elevenlabs.conversational_ai.conversation import Conversation

            # Create conversation
            conv = Conversation(
                client=MagicMock(),
                agent_id="test_agent_123",
                user_id="user_456",
            )

            assert conv.agent_id == "test_agent_123"
            assert conv.user_id == "user_456"

            # Start session - this creates the span
            conv.start_session()

            # End the session to finish the span
            conv.end_session()

            # Now check spans
            spans = in_memory_span_exporter.get_finished_spans()
            assert len(spans) == 1

            span = spans[0]
            assert span.name == "ElevenLabs.Conversation"

            attrs = dict(span.attributes or {})
            assert attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "CHAIN"
            assert attrs.get(ELEVENLABS_AGENT_ID) == "test_agent_123"
            assert attrs.get(SpanAttributes.USER_ID) == "user_456"

        finally:
            instrumentor.uninstrument()

    def test_conversation_start_session_adds_event(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_conversation_module: None,
    ) -> None:
        """Test that start_session adds an event to the span."""
        instrumentor = ElevenLabsInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            from elevenlabs.conversational_ai.conversation import Conversation

            conv = Conversation(
                client=MagicMock(),
                agent_id="agent_123",
            )

            conv.start_session()
            assert conv._session_started

            conv.end_session()

            spans = in_memory_span_exporter.get_finished_spans()
            assert len(spans) == 1

            # Check for session_started event
            events = spans[0].events
            assert len(events) >= 1
            event_names = [e.name for e in events]
            assert "session_started" in event_names

        finally:
            instrumentor.uninstrument()

    def test_conversation_wait_for_session_end_captures_id(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_conversation_module: None,
    ) -> None:
        """Test that wait_for_session_end captures conversation ID."""
        instrumentor = ElevenLabsInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            from elevenlabs.conversational_ai.conversation import Conversation

            conv = Conversation(
                client=MagicMock(),
                agent_id="agent_123",
            )

            conv.start_session()
            conversation_id = conv.wait_for_session_end()

            assert conversation_id == "test_conversation_123"

            spans = in_memory_span_exporter.get_finished_spans()
            assert len(spans) == 1

            attrs = dict(spans[0].attributes or {})
            assert attrs.get(ELEVENLABS_CONVERSATION_ID) == "test_conversation_123"

        finally:
            instrumentor.uninstrument()

    def test_conversation_without_user_id(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_conversation_module: None,
    ) -> None:
        """Test conversation without user_id."""
        instrumentor = ElevenLabsInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            from elevenlabs.conversational_ai.conversation import Conversation

            conv = Conversation(
                client=MagicMock(),
                agent_id="agent_only",
            )

            # start_session() creates the span
            conv.start_session()
            conv.end_session()

            spans = in_memory_span_exporter.get_finished_spans()
            assert len(spans) == 1

            attrs = dict(spans[0].attributes or {})
            assert attrs.get(ELEVENLABS_AGENT_ID) == "agent_only"
            # user_id should not be present
            assert SpanAttributes.USER_ID not in attrs or attrs.get(SpanAttributes.USER_ID) is None

        finally:
            instrumentor.uninstrument()


class TestConversationAttributes:
    """Test conversation attribute extraction."""

    def test_conversation_provider_attributes(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_conversation_module: None,
    ) -> None:
        """Test that provider attributes are set correctly."""
        instrumentor = ElevenLabsInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            from elevenlabs.conversational_ai.conversation import Conversation

            conv = Conversation(
                client=MagicMock(),
                agent_id="test_agent",
            )
            # start_session() creates the span
            conv.start_session()
            conv.end_session()

            spans = in_memory_span_exporter.get_finished_spans()
            attrs = dict(spans[0].attributes or {})

            assert attrs.get(SpanAttributes.LLM_PROVIDER) == "elevenlabs"
            assert attrs.get(SpanAttributes.LLM_SYSTEM) == "elevenlabs"

        finally:
            instrumentor.uninstrument()
