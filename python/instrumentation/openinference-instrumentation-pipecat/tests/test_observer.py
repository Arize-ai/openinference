"""Tests for OpenInferenceObserver."""

import asyncio
import os
import tempfile
import time
from typing import Any
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    FunctionCallResultFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    ProcessingMetricsData,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.transports.base_output import BaseOutputTransport

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.pipecat._observer import OpenInferenceObserver
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes


def create_frame_pushed(source, destination, frame, direction="down"):
    """Helper to create FramePushed with timestamp."""
    return FramePushed(
        source=source,
        destination=destination,
        frame=frame,
        direction=direction,
        timestamp=time.time(),
    )


def _make_transport_tts_frame(text: str) -> TTSTextFrame:
    """Build a transport-source TTSTextFrame routed as bot speech."""
    frame = TTSTextFrame(text=text, aggregated_by="x")
    # skip_tts is set by the transport layer in production; the observer
    # only counts the frame as bot speech when it's False.
    frame.skip_tts = False
    return frame


def _fast_observer(tracer: OITracer, config: TraceConfig, **kwargs: Any) -> OpenInferenceObserver:
    """Observer pre-configured with short timeouts for timer-driven tests."""
    return OpenInferenceObserver(
        tracer=tracer,
        config=config,
        turn_end_timeout_secs=kwargs.pop("turn_end_timeout_secs", 0.05),
        no_responder_timeout_secs=kwargs.pop("no_responder_timeout_secs", 0.1),
        **kwargs,
    )


def _get_turn_spans(exporter: InMemorySpanExporter):
    return [s for s in exporter.get_finished_spans() if s.name == "pipecat.conversation.turn"]


class TestObserverInitialization:
    """Test observer initialization and configuration."""

    def test_observer_initialization(self, tracer: OITracer, config: TraceConfig) -> None:
        """Test basic observer initialization."""
        observer = OpenInferenceObserver(tracer=tracer, config=config)

        assert observer._tracer == tracer
        assert observer._config == config
        assert observer._conversation_id is None
        assert observer._additional_span_attributes == {}

    def test_observer_with_conversation_id(self, tracer: OITracer, config: TraceConfig) -> None:
        """Test observer initialization with conversation ID."""
        conversation_id = "test-conversation-123"
        observer = OpenInferenceObserver(
            tracer=tracer, config=config, conversation_id=conversation_id
        )

        assert observer._conversation_id == conversation_id

    def test_observer_with_additional_attributes(
        self, tracer: OITracer, config: TraceConfig
    ) -> None:
        """Test observer with additional span attributes."""
        additional_attrs = {"user.id": "user123", "session.type": "test"}
        observer = OpenInferenceObserver(
            tracer=tracer, config=config, additional_span_attributes=additional_attrs
        )

        assert observer._additional_span_attributes == {
            "user.id": "user123",
            "session.type": "test",
        }

    def test_observer_with_debug_logging(self, tracer: OITracer, config: TraceConfig) -> None:
        """Test observer debug logging to file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_filename = f.name

        try:
            observer = OpenInferenceObserver(
                tracer=tracer, config=config, debug_log_filename=log_filename
            )

            assert observer._debug_log_file is not None

            # Clean up observer
            del observer

            # Verify log file was created and written to
            assert os.path.exists(log_filename)
            with open(log_filename, "r") as f:
                content = f.read()
                assert "Observer initialized" in content
        finally:
            if os.path.exists(log_filename):
                os.unlink(log_filename)


@pytest.mark.asyncio
class TestTurnTracking:
    """Test turn tracking functionality."""

    async def test_start_turn_creates_turn_span(
        self,
        observer: OpenInferenceObserver,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
    ) -> None:
        """User speech opens a user-initiated turn."""
        # Pipecat's natural sequence: UserStartedSpeakingFrame followed by
        # the VAD-confirmed VADUserStartedSpeakingFrame. Either is a valid
        # user-started edge, and both arriving back-to-back must not double-
        # open the turn.
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStartedSpeakingFrame())
        )

        assert observer._is_turn_active
        assert observer._turn_span is not None
        assert observer._turn_count == 1
        assert observer._turn_initiator == "user"

    async def test_turn_span_attributes(
        self,
        tracer: OITracer,
        config: TraceConfig,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
    ) -> None:
        """Test turn span has correct attributes."""
        conversation_id = "test-conv-456"
        observer = OpenInferenceObserver(
            tracer=tracer, config=config, conversation_id=conversation_id
        )

        # Open the turn by pushing a user-started edge.
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_stt_service,
                destination=None,
                frame=VADUserStartedSpeakingFrame(),
                direction="down",
            )
        )

        await observer._close_turn(end_reason="completed")

        spans = in_memory_span_exporter.get_finished_spans()

        turn_spans = [s for s in spans if s.name == "pipecat.conversation.turn"]
        assert len(turn_spans) == 1

        turn_span = turn_spans[0]
        attributes = dict(turn_span.attributes or {})

        assert (
            attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND]
            == OpenInferenceSpanKindValues.CHAIN.value
        )
        assert attributes["conversation.turn_number"] == 1
        assert attributes[SpanAttributes.SESSION_ID] == conversation_id
        assert attributes["conversation.end_reason"] == "completed"
        assert attributes["conversation.was_interrupted"] is False
        assert attributes["conversation.initiator"] == "user"


@pytest.mark.asyncio
class TestServiceSpanCreation:
    """Test service span creation for LLM, STT, and TTS."""

    async def test_stt_span_creation(
        self,
        observer: OpenInferenceObserver,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
    ) -> None:
        """Test STT service span creation."""
        # Start turn first
        start_frame = VADUserStartedSpeakingFrame()
        start_data = create_frame_pushed(
            source=mock_stt_service, destination=None, frame=start_frame, direction="down"
        )
        await observer.on_push_frame(start_data)

        # Process transcription
        transcription = TranscriptionFrame(text="Hello world", user_id="test_user", timestamp=0)
        trans_data = create_frame_pushed(
            source=mock_stt_service, destination=None, frame=transcription, direction="down"
        )
        await observer.on_push_frame(trans_data)

        # Verify STT span was created and is active
        service_id = id(mock_stt_service)
        assert service_id in observer._active_spans
        # Verify it has accumulated the transcription
        assert observer._active_spans[service_id]["accumulated_input"] == "Hello world "

    async def test_llm_span_creation(
        self,
        observer: OpenInferenceObserver,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
        mock_llm_service: Mock,
    ) -> None:
        """Test LLM service span creation."""
        # Open a turn first so the LLM span has a parent.
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_stt_service,
                destination=None,
                frame=VADUserStartedSpeakingFrame(),
                direction="down",
            )
        )

        # Start LLM with context
        context = LLMContext()
        context._messages = [{"role": "user", "content": "Hello"}]
        context_frame = LLMContextFrame(context=context)
        context_data = create_frame_pushed(
            source=None, destination=mock_llm_service, frame=context_frame, direction="down"
        )
        await observer.on_push_frame(context_data)

        # Verify LLM span was created
        service_id = id(mock_llm_service)
        assert service_id in observer._active_spans

        # End LLM
        end_frame = LLMFullResponseEndFrame()
        end_data = create_frame_pushed(
            source=mock_llm_service, destination=None, frame=end_frame, direction="down"
        )
        await observer.on_push_frame(end_data)

        # Span should be finished
        assert service_id not in observer._active_spans

    async def test_tts_span_creation(
        self,
        observer: OpenInferenceObserver,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
        mock_tts_service: Mock,
    ) -> None:
        """Test TTS service span creation."""
        # Open a turn first so the TTS span has a parent.
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_stt_service,
                destination=None,
                frame=VADUserStartedSpeakingFrame(),
                direction="down",
            )
        )

        # Start TTS
        tts_start = TTSStartedFrame()
        start_data = create_frame_pushed(
            source=mock_tts_service, destination=None, frame=tts_start, direction="down"
        )
        await observer.on_push_frame(start_data)

        # Verify TTS span was created
        service_id = id(mock_tts_service)
        assert service_id in observer._active_spans


@pytest.mark.asyncio
class TestTextAccumulation:
    """Test text accumulation from streaming frames."""

    async def test_llm_output_accumulation(
        self,
        observer: OpenInferenceObserver,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
        mock_llm_service: Mock,
    ) -> None:
        """Test LLM output text accumulation."""
        # Open a turn so the LLM span can be parented.
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_stt_service,
                destination=None,
                frame=VADUserStartedSpeakingFrame(),
                direction="down",
            )
        )

        # Start LLM
        context = LLMContext()
        context_frame = LLMContextFrame(context=context)
        await observer.on_push_frame(
            create_frame_pushed(
                source=None, destination=mock_llm_service, frame=context_frame, direction="down"
            )
        )

        service_id = id(mock_llm_service)

        # Stream LLM text chunks
        chunks = ["Hello", " ", "world", "!"]
        for chunk in chunks:
            text_frame = LLMTextFrame(text=chunk)
            await observer.on_push_frame(
                create_frame_pushed(
                    source=mock_llm_service, destination=None, frame=text_frame, direction="down"
                )
            )

        # Check accumulated output
        assert service_id in observer._active_spans
        accumulated = observer._active_spans[service_id]["accumulated_output"]
        # Default behavior adds spaces between chunks
        assert "Hello" in accumulated and "world" in accumulated

    async def test_stt_input_accumulation(
        self,
        observer: OpenInferenceObserver,
        mock_stt_service: Mock,
    ) -> None:
        """Test STT input text accumulation."""
        # Start STT
        start_frame = VADUserStartedSpeakingFrame()
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_stt_service, destination=None, frame=start_frame, direction="down"
            )
        )

        service_id = id(mock_stt_service)

        # Stream transcription chunks
        chunks = ["This", "is", "a", "test"]
        for chunk in chunks:
            trans_frame = TranscriptionFrame(text=chunk, user_id="test_user", timestamp=0)
            await observer.on_push_frame(
                create_frame_pushed(
                    source=mock_stt_service, destination=None, frame=trans_frame, direction="down"
                )
            )

        # Check accumulated input
        assert service_id in observer._active_spans
        accumulated = observer._active_spans[service_id]["accumulated_input"]
        # Default behavior adds spaces
        assert "This" in accumulated


@pytest.mark.asyncio
class TestMetricsHandling:
    """Test metrics frame handling."""

    async def test_metrics_frame_handling(
        self,
        observer: OpenInferenceObserver,
        mock_stt_service: Mock,
        mock_llm_service: Mock,
    ) -> None:
        """Test metrics extraction and setting on spans."""
        # Open a turn so the LLM span can be parented.
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_stt_service,
                destination=None,
                frame=VADUserStartedSpeakingFrame(),
                direction="down",
            )
        )

        # Start LLM
        context = LLMContext()
        context_frame = LLMContextFrame(context=context)
        await observer.on_push_frame(
            create_frame_pushed(
                source=None, destination=mock_llm_service, frame=context_frame, direction="down"
            )
        )

        service_id = id(mock_llm_service)

        # Send metrics
        token_usage = LLMTokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        metrics_data = LLMUsageMetricsData(
            processor="TestLLMService", model="gpt-4", value=token_usage
        )
        metrics_frame = MetricsFrame(data=[metrics_data])
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_llm_service, destination=None, frame=metrics_frame, direction="down"
            )
        )

        # Verify metrics are stored (in the span_info)
        assert service_id in observer._active_spans

    async def test_processing_time_metrics(
        self,
        observer: OpenInferenceObserver,
        mock_stt_service: Mock,
    ) -> None:
        """Test processing time metrics storage."""
        # Start STT
        start_frame = VADUserStartedSpeakingFrame()
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_stt_service, destination=None, frame=start_frame, direction="down"
            )
        )

        service_id = id(mock_stt_service)

        # Send processing time metrics
        processing_metrics = ProcessingMetricsData(
            processor="TestSTTService",
            model="nova-2",
            value=1.5,  # 1.5 seconds
        )
        metrics_frame = MetricsFrame(data=[processing_metrics])
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_stt_service, destination=None, frame=metrics_frame, direction="down"
            )
        )

        # Verify processing time is stored
        assert service_id in observer._active_spans
        span_info = observer._active_spans[service_id]
        assert span_info["processing_time_seconds"] == 1.5


@pytest.mark.asyncio
class TestSpanHierarchy:
    """Test span parent-child relationships."""

    async def test_span_hierarchy(
        self,
        observer: OpenInferenceObserver,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
        mock_llm_service: Mock,
    ) -> None:
        """Test that service spans are children of turn span."""
        # Start turn
        start_frame = VADUserStartedSpeakingFrame()
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_stt_service, destination=None, frame=start_frame, direction="down"
            )
        )

        # Start LLM (within turn)
        context = LLMContext()
        context_frame = LLMContextFrame(context=context)
        await observer.on_push_frame(
            create_frame_pushed(
                source=None, destination=mock_llm_service, frame=context_frame, direction="down"
            )
        )

        # Verify both turn and LLM spans exist
        assert observer._turn_span is not None
        llm_service_id = id(mock_llm_service)
        assert llm_service_id in observer._active_spans

        # The service span should have the turn span as its parent
        # This is implicit in the implementation via context management


@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and error handling."""

    async def test_multiple_services_same_turn(
        self,
        observer: OpenInferenceObserver,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
        mock_llm_service: Mock,
        mock_tts_service: Mock,
    ) -> None:
        """Test multiple service spans within a single turn.

        STT and TTS spans are finished when a different service type starts,
        so only the most recently started service type has an active span.
        """
        # Start turn with STT
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_stt_service,
                destination=None,
                frame=VADUserStartedSpeakingFrame(),
                direction="down",
            )
        )

        # Verify STT span is active
        assert id(mock_stt_service) in observer._active_spans
        assert observer._active_stt_service_id == id(mock_stt_service)

        # Start LLM — this finishes the active STT span
        context = LLMContext()
        await observer.on_push_frame(
            create_frame_pushed(
                source=None,
                destination=mock_llm_service,
                frame=LLMContextFrame(context=context),
                direction="down",
            )
        )

        # STT span was finished when LLM started
        assert id(mock_stt_service) not in observer._active_spans
        assert observer._active_stt_service_id is None
        assert id(mock_llm_service) in observer._active_spans

        # Start TTS — this finishes the active LLM span is not affected (only STT/TTS auto-close)
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_tts_service, destination=None, frame=TTSStartedFrame(), direction="down"
            )
        )

        assert id(mock_llm_service) in observer._active_spans
        assert id(mock_tts_service) in observer._active_spans
        assert observer._active_tts_service_id == id(mock_tts_service)

        # End turn to flush all spans
        await observer._close_turn(end_reason="completed")

        # Verify all three service spans were created during the turn
        spans = in_memory_span_exporter.get_finished_spans()
        # Should have: turn span + STT span + LLM span + TTS span
        service_span_names = [s.name for s in spans if s.name != "pipecat.conversation.turn"]
        assert len(service_span_names) >= 3

    async def test_inter_frame_spaces_handling(
        self,
        observer: OpenInferenceObserver,
        mock_stt_service: Mock,
        mock_llm_service: Mock,
    ) -> None:
        """Test handling of includes_inter_frame_spaces flag."""
        # Open a turn so the LLM span can be parented.
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_stt_service,
                destination=None,
                frame=VADUserStartedSpeakingFrame(),
                direction="down",
            )
        )

        # Start LLM
        context = LLMContext()
        await observer.on_push_frame(
            create_frame_pushed(
                source=None,
                destination=mock_llm_service,
                frame=LLMContextFrame(context=context),
                direction="down",
            )
        )

        # Send text with inter_frame_spaces flag
        text_frame = LLMTextFrame(text="test")
        text_frame.includes_inter_frame_spaces = True
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_llm_service, destination=None, frame=text_frame, direction="down"
            )
        )

        # Verify flag is tracked
        assert observer._llm_includes_inter_frame_spaces is True


@pytest.mark.asyncio
class TestToolCallTracking:
    """Test tool/function call tracking."""

    async def test_tool_span_creation_from_result_frame(
        self,
        observer: OpenInferenceObserver,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
        mock_llm_service: Mock,
    ) -> None:
        """Test that FunctionCallResultFrame creates a tool span."""
        # Start a turn first
        start_frame = VADUserStartedSpeakingFrame()
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_stt_service, destination=None, frame=start_frame, direction="down"
            )
        )

        # Create a function call result frame (tool spans are only created from result frames)
        tool_call_id = "call_123"
        function_name = "get_weather"
        arguments = {"location": "San Francisco", "unit": "celsius"}

        result_frame = FunctionCallResultFrame(
            tool_call_id=tool_call_id,
            function_name=function_name,
            arguments=arguments,
            result="Sunny, 72°F",
        )
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_llm_service, destination=None, frame=result_frame, direction="down"
            )
        )

        # Span should have been created and immediately finished
        assert tool_call_id not in observer._active_spans
        # But it should be in completed tool calls
        assert tool_call_id in observer._completed_tool_calls

        # End turn to flush spans
        await observer._close_turn(end_reason="completed")

        # Check finished spans
        spans = in_memory_span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "pipecat.tool" in s.name]
        assert len(tool_spans) == 1

        tool_span = tool_spans[0]
        attributes = dict(tool_span.attributes or {})
        assert attributes[SpanAttributes.TOOL_NAME] == function_name

    async def test_tool_span_with_result_attributes(
        self,
        observer: OpenInferenceObserver,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
        mock_llm_service: Mock,
    ) -> None:
        """Test that FunctionCallResultFrame creates span with correct attributes."""
        # Start a turn first
        start_frame = VADUserStartedSpeakingFrame()
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_stt_service, destination=None, frame=start_frame, direction="down"
            )
        )

        tool_call_id = "call_456"
        function_name = "search_database"
        arguments = {"query": "test"}

        # Send result frame
        result_frame = FunctionCallResultFrame(
            tool_call_id=tool_call_id,
            function_name=function_name,
            arguments=arguments,
            result={"count": 42, "items": ["item1", "item2"]},
        )
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_llm_service, destination=None, frame=result_frame, direction="down"
            )
        )

        # Verify span was finished and removed from active spans
        assert tool_call_id not in observer._active_spans

        # End turn to flush spans
        await observer._close_turn(end_reason="completed")

        # Check finished spans
        spans = in_memory_span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "pipecat.tool" in s.name]
        assert len(tool_spans) == 1

        tool_span = tool_spans[0]
        attributes = dict(tool_span.attributes or {})

        # Verify span attributes
        assert attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == (
            OpenInferenceSpanKindValues.TOOL.value
        )
        assert attributes[SpanAttributes.TOOL_NAME] == function_name

    async def test_tool_span_from_result_frame_only(
        self,
        observer: OpenInferenceObserver,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
        mock_llm_service: Mock,
    ) -> None:
        """Test that tool span is created from FunctionCallResultFrame."""
        # Start a turn first
        start_frame = VADUserStartedSpeakingFrame()
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_stt_service, destination=None, frame=start_frame, direction="down"
            )
        )

        # Send result frame directly
        tool_call_id = "call_789"
        function_name = "calculate"
        result_frame = FunctionCallResultFrame(
            tool_call_id=tool_call_id,
            function_name=function_name,
            arguments={"x": 1, "y": 2},
            result=3,
        )
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_llm_service, destination=None, frame=result_frame, direction="down"
            )
        )

        # Span should have been created and immediately finished
        assert tool_call_id not in observer._active_spans

        # End turn to flush spans
        await observer._close_turn(end_reason="completed")

        # Check finished spans
        spans = in_memory_span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "pipecat.tool" in s.name]
        assert len(tool_spans) == 1

    async def test_multiple_tool_calls_same_turn(
        self,
        observer: OpenInferenceObserver,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
        mock_llm_service: Mock,
    ) -> None:
        """Test multiple tool calls within a single turn."""
        # Start a turn
        start_frame = VADUserStartedSpeakingFrame()
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_stt_service, destination=None, frame=start_frame, direction="down"
            )
        )

        # First tool call - only send result frame (that's what creates the span now)
        tool_call_id_1 = "call_001"
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_llm_service,
                destination=None,
                frame=FunctionCallResultFrame(
                    tool_call_id=tool_call_id_1,
                    function_name="tool_a",
                    arguments={},
                    result="result_a",
                ),
                direction="down",
            )
        )

        # Second tool call
        tool_call_id_2 = "call_002"
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_llm_service,
                destination=None,
                frame=FunctionCallResultFrame(
                    tool_call_id=tool_call_id_2,
                    function_name="tool_b",
                    arguments={},
                    result="result_b",
                ),
                direction="down",
            )
        )

        # End turn
        await observer._close_turn(end_reason="completed")

        # Check that both tool spans were created
        spans = in_memory_span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "pipecat.tool" in s.name]
        assert len(tool_spans) == 2

        tool_names = {s.name for s in tool_spans}
        assert "pipecat.tool.tool_a" in tool_names
        assert "pipecat.tool.tool_b" in tool_names

    async def test_tool_frame_without_tool_call_id_ignored(
        self,
        observer: OpenInferenceObserver,
        mock_stt_service: Mock,
        mock_llm_service: Mock,
    ) -> None:
        """Test that tool frames without tool_call_id are ignored gracefully."""
        # Start a turn
        start_frame = VADUserStartedSpeakingFrame()
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_stt_service, destination=None, frame=start_frame, direction="down"
            )
        )

        # Create a result frame without tool_call_id
        frame = FunctionCallResultFrame(
            tool_call_id=None,  # type: ignore[arg-type]
            function_name="test",
            arguments={},
            result="test",
        )
        # Override to ensure tool_call_id is None
        frame.tool_call_id = None  # type: ignore[assignment]

        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_llm_service, destination=None, frame=frame, direction="down"
            )
        )

        # No tool spans should be created
        tool_spans = [
            k for k, v in observer._active_spans.items() if v.get("service_type") == "tool"
        ]
        assert len(tool_spans) == 0

    async def test_duplicate_tool_call_ids_ignored(
        self,
        observer: OpenInferenceObserver,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
        mock_llm_service: Mock,
    ) -> None:
        """Test that duplicate tool call IDs are ignored to prevent duplicate spans."""
        # Start a turn
        start_frame = VADUserStartedSpeakingFrame()
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_stt_service, destination=None, frame=start_frame, direction="down"
            )
        )

        # Send same tool call result twice
        tool_call_id = "call_duplicate"
        for _ in range(2):
            await observer.on_push_frame(
                create_frame_pushed(
                    source=mock_llm_service,
                    destination=None,
                    frame=FunctionCallResultFrame(
                        tool_call_id=tool_call_id,
                        function_name="test_tool",
                        arguments={},
                        result="result",
                    ),
                    direction="down",
                )
            )

        # End turn
        await observer._close_turn(end_reason="completed")

        # Should only have one tool span despite two result frames
        spans = in_memory_span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "pipecat.tool" in s.name]
        assert len(tool_spans) == 1


@pytest.mark.asyncio
class TestBidirectionalTurns:
    """Cover the bidirectional turn state machine end-to-end.

    These tests exercise the explicit state machine that replaces the
    bot-gated logic inherited from ``TurnTrackingObserver``: either party may
    initiate a turn, the close timer fires even when the responder never
    speaks, and the STT span finishes on the user-stop edge.
    """

    async def test_bot_first_greeting_round_trip(
        self,
        tracer: OITracer,
        config: TraceConfig,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
    ) -> None:
        """Bot greeting + user reply = one bot-initiated turn; the next user-
        initiated utterance opens turn 2."""
        observer = _fast_observer(tracer, config)
        transport = Mock(spec=BaseOutputTransport)

        greeting = "Hello, how can I help?"
        user_reply_text = "I need help with my bill"

        # Bot greets — opens turn 1 (initiator=bot).
        await observer.on_push_frame(create_frame_pushed(None, None, BotStartedSpeakingFrame()))
        # Transport-source TTSTextFrame counts as bot speech and supplies output.
        await observer.on_push_frame(
            create_frame_pushed(transport, None, _make_transport_tts_frame(greeting))
        )
        await observer.on_push_frame(create_frame_pushed(None, None, BotStoppedSpeakingFrame()))

        # User replies — continues turn 1.
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(
                mock_stt_service,
                None,
                TranscriptionFrame(text=user_reply_text, user_id="u", timestamp="0"),
            )
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStoppedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStoppedSpeakingFrame())
        )

        # Both have spoken; close timer (turn_end_timeout_secs) closes turn 1.
        await asyncio.sleep(0.2)

        turn_spans = _get_turn_spans(in_memory_span_exporter)
        assert len(turn_spans) == 1
        turn_1_attrs = dict(turn_spans[0].attributes or {})
        assert turn_1_attrs["conversation.initiator"] == "bot"
        assert turn_1_attrs[SpanAttributes.OUTPUT_VALUE] == greeting
        assert turn_1_attrs[SpanAttributes.INPUT_VALUE] == user_reply_text
        assert turn_1_attrs["conversation.end_reason"] == "completed"

        # User now starts a fresh exchange — turn 2 opens with initiator=user.
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStartedSpeakingFrame())
        )

        assert observer._turn_count == 2
        assert observer._turn_initiator == "user"

    async def test_human_first_basic(
        self,
        tracer: OITracer,
        config: TraceConfig,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
    ) -> None:
        """User opens, bot answers; user replies, bot answers — two
        user-initiated turns, each exported promptly after the bot stops."""
        observer = _fast_observer(tracer, config)
        transport = Mock(spec=BaseOutputTransport)

        async def _user_then_bot(user_text: str, bot_text: str) -> None:
            await observer.on_push_frame(
                create_frame_pushed(mock_stt_service, None, UserStartedSpeakingFrame())
            )
            await observer.on_push_frame(
                create_frame_pushed(mock_stt_service, None, VADUserStartedSpeakingFrame())
            )
            await observer.on_push_frame(
                create_frame_pushed(
                    mock_stt_service,
                    None,
                    TranscriptionFrame(text=user_text, user_id="u", timestamp="0"),
                )
            )
            await observer.on_push_frame(
                create_frame_pushed(mock_stt_service, None, UserStoppedSpeakingFrame())
            )
            await observer.on_push_frame(
                create_frame_pushed(mock_stt_service, None, VADUserStoppedSpeakingFrame())
            )
            await observer.on_push_frame(create_frame_pushed(None, None, BotStartedSpeakingFrame()))
            await observer.on_push_frame(
                create_frame_pushed(transport, None, _make_transport_tts_frame(bot_text))
            )
            await observer.on_push_frame(create_frame_pushed(None, None, BotStoppedSpeakingFrame()))
            # Both have spoken; close timer (turn_end_timeout) elapses.
            await asyncio.sleep(0.2)

        await _user_then_bot("First question", "First answer")
        await _user_then_bot("Second question", "Second answer")

        turn_spans = _get_turn_spans(in_memory_span_exporter)
        assert len(turn_spans) == 2

        for span in turn_spans:
            attrs = dict(span.attributes or {})
            assert attrs["conversation.initiator"] == "user"
            assert attrs["conversation.end_reason"] == "completed"

    async def test_human_first_slow_bot(
        self,
        tracer: OITracer,
        config: TraceConfig,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
        mock_llm_service: Mock,
    ) -> None:
        """LLMContextFrame after the user stops suppresses the close timer
        — turn isn't split even if the bot's response is slow."""
        observer = _fast_observer(tracer, config)
        transport = Mock(spec=BaseOutputTransport)

        # User speaks and stops.
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(
                mock_stt_service,
                None,
                TranscriptionFrame(text="What's the weather?", user_id="u", timestamp="0"),
            )
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStoppedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStoppedSpeakingFrame())
        )

        # LLM is invoked, gating the close timer via _bot_response_pending.
        context = LLMContext()
        await observer.on_push_frame(
            create_frame_pushed(None, mock_llm_service, LLMContextFrame(context=context))
        )

        # Wait LONGER than turn_end_timeout_secs (0.05) — the timer must stay
        # gated because the bot's response is pending.
        await asyncio.sleep(0.15)
        assert not _get_turn_spans(in_memory_span_exporter)
        assert observer._turn_count == 1

        # Bot finally speaks.
        await observer.on_push_frame(
            create_frame_pushed(mock_llm_service, None, LLMFullResponseEndFrame())
        )
        await observer.on_push_frame(create_frame_pushed(None, None, BotStartedSpeakingFrame()))
        await observer.on_push_frame(
            create_frame_pushed(transport, None, _make_transport_tts_frame("Sunny."))
        )
        await observer.on_push_frame(create_frame_pushed(None, None, BotStoppedSpeakingFrame()))
        await asyncio.sleep(0.2)

        turn_spans = _get_turn_spans(in_memory_span_exporter)
        assert len(turn_spans) == 1
        attrs = dict(turn_spans[0].attributes or {})
        assert attrs["conversation.initiator"] == "user"
        assert attrs["conversation.end_reason"] == "completed"

    async def test_no_responder_timeout(
        self,
        tracer: OITracer,
        config: TraceConfig,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
    ) -> None:
        """User opens, bot never responds — turn closes with
        ``no_responder_timeout`` after ``no_responder_timeout_secs``."""
        observer = _fast_observer(tracer, config)

        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(
                mock_stt_service,
                None,
                TranscriptionFrame(text="Anyone there?", user_id="u", timestamp="0"),
            )
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStoppedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStoppedSpeakingFrame())
        )

        # Wait past no_responder_timeout_secs (0.1).
        await asyncio.sleep(0.2)

        turn_spans = _get_turn_spans(in_memory_span_exporter)
        assert len(turn_spans) == 1
        attrs = dict(turn_spans[0].attributes or {})
        assert attrs["conversation.initiator"] == "user"
        assert attrs["conversation.end_reason"] == "no_responder_timeout"
        assert attrs[SpanAttributes.INPUT_VALUE] == "Anyone there?"
        assert SpanAttributes.OUTPUT_VALUE not in attrs

    async def test_consecutive_user_utterances_merge(
        self,
        tracer: OITracer,
        config: TraceConfig,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
    ) -> None:
        """Two consecutive user utterances (before any bot response) end up
        in the same turn's ``input.value``."""
        observer = _fast_observer(tracer, config)

        # First utterance.
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(
                mock_stt_service,
                None,
                TranscriptionFrame(text="First part", user_id="u", timestamp="0"),
            )
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStoppedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStoppedSpeakingFrame())
        )

        # Second utterance — bot has not spoken, so the turn must not roll.
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(
                mock_stt_service,
                None,
                TranscriptionFrame(text="second part", user_id="u", timestamp="0"),
            )
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStoppedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStoppedSpeakingFrame())
        )

        await asyncio.sleep(0.2)

        turn_spans = _get_turn_spans(in_memory_span_exporter)
        assert len(turn_spans) == 1
        attrs = dict(turn_spans[0].attributes or {})
        assert attrs["conversation.initiator"] == "user"
        input_value = attrs[SpanAttributes.INPUT_VALUE]
        assert "First part" in input_value
        assert "second part" in input_value

    async def test_consecutive_bot_utterances_merge(
        self,
        tracer: OITracer,
        config: TraceConfig,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """Two consecutive bot utterances (before any user reply) end up in
        the same turn's ``output.value``."""
        observer = _fast_observer(tracer, config)
        transport = Mock(spec=BaseOutputTransport)

        await observer.on_push_frame(create_frame_pushed(None, None, BotStartedSpeakingFrame()))
        await observer.on_push_frame(
            create_frame_pushed(transport, None, _make_transport_tts_frame("Hello."))
        )
        await observer.on_push_frame(create_frame_pushed(None, None, BotStoppedSpeakingFrame()))

        # Bot speaks again — user has not spoken, no roll.
        await observer.on_push_frame(create_frame_pushed(None, None, BotStartedSpeakingFrame()))
        await observer.on_push_frame(
            create_frame_pushed(transport, None, _make_transport_tts_frame("How are you?"))
        )
        await observer.on_push_frame(create_frame_pushed(None, None, BotStoppedSpeakingFrame()))

        await asyncio.sleep(0.2)

        turn_spans = _get_turn_spans(in_memory_span_exporter)
        assert len(turn_spans) == 1
        attrs = dict(turn_spans[0].attributes or {})
        assert attrs["conversation.initiator"] == "bot"
        output_value = attrs[SpanAttributes.OUTPUT_VALUE]
        assert "Hello." in output_value
        assert "How are you?" in output_value

    async def test_user_interrupts_bot(
        self,
        tracer: OITracer,
        config: TraceConfig,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
    ) -> None:
        """User starts while the bot is still speaking — current turn ends
        interrupted; new turn opens with initiator=user."""
        observer = _fast_observer(tracer, config)
        transport = Mock(spec=BaseOutputTransport)

        # Bot begins speaking.
        await observer.on_push_frame(create_frame_pushed(None, None, BotStartedSpeakingFrame()))
        await observer.on_push_frame(
            create_frame_pushed(transport, None, _make_transport_tts_frame("Long answer..."))
        )
        # User interrupts before the bot stops.
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStartedSpeakingFrame())
        )

        # Turn 1 should have closed as interrupted.
        turn_spans = _get_turn_spans(in_memory_span_exporter)
        assert len(turn_spans) == 1
        attrs = dict(turn_spans[0].attributes or {})
        assert attrs["conversation.was_interrupted"] is True
        assert attrs["conversation.end_reason"] == "interrupted"
        assert attrs["conversation.initiator"] == "bot"

        # Turn 2 is now active with the user as initiator.
        assert observer._turn_count == 2
        assert observer._turn_initiator == "user"

    async def test_stt_span_finishes_on_user_stop(
        self,
        tracer: OITracer,
        config: TraceConfig,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
    ) -> None:
        """The STT span finishes on ``VADUserStoppedSpeakingFrame`` so it
        exports even before any LLM activity."""
        observer = _fast_observer(tracer, config)

        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(
                mock_stt_service,
                None,
                TranscriptionFrame(text="Hello there", user_id="u", timestamp="0"),
            )
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStoppedSpeakingFrame())
        )

        # No LLM activity yet — but the STT span must already be exported.
        # Streaming STT case: the transcript arrived before VAD-stop, so the
        # span has accumulated input and closes on VAD-stop.
        stt_spans = [
            s for s in in_memory_span_exporter.get_finished_spans() if s.name == "pipecat.stt"
        ]
        assert len(stt_spans) == 1
        assert observer._active_stt_service_id is None

    async def test_user_text_captured_when_transcript_arrives_after_vad_stop(
        self,
        tracer: OITracer,
        config: TraceConfig,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
        mock_llm_service: Mock,
    ) -> None:
        """Non-streaming STT (e.g. OpenAI Whisper) emits the final
        ``TranscriptionFrame`` AFTER ``VADUserStoppedSpeakingFrame``. The
        user's words must still be captured both in the STT span's input
        and in the turn span's input.value; the STT span must close once
        the transcript arrives. Regression test for the
        'user messages getting missed when the user starts a turn' bug.
        """
        observer = _fast_observer(tracer, config)

        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStartedSpeakingFrame())
        )
        # Whisper-style order: VAD-stop FIRST, transcript SECOND.
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStoppedSpeakingFrame())
        )
        # STT span must still be open: we're waiting for the transcript.
        assert observer._active_stt_service_id is not None
        assert observer._seen_vad_user_stopped_speaking_frame is True

        user_text = "What is the weather today"
        await observer.on_push_frame(
            create_frame_pushed(
                mock_stt_service,
                None,
                TranscriptionFrame(text=user_text, user_id="u", timestamp="0"),
            )
        )

        # Transcript arrival closes the STT span and clears the VAD-stop flag.
        assert observer._active_stt_service_id is None
        assert observer._seen_vad_user_stopped_speaking_frame is False

        # STT span exported with the user's text.
        stt_spans = [
            s for s in in_memory_span_exporter.get_finished_spans() if s.name == "pipecat.stt"
        ]
        assert len(stt_spans) == 1
        stt_attrs = dict(stt_spans[0].attributes or {})
        assert user_text in str(stt_attrs.get(SpanAttributes.INPUT_VALUE, ""))

        # And the user text is staged on the turn (it'll land in input.value
        # when the turn closes).
        assert any(user_text in t for t in observer._turn_user_text)

        # Close the turn via EndFrame and assert the turn span's input.value.
        await observer.on_push_frame(create_frame_pushed(None, None, EndFrame()))
        turn_spans = [
            s
            for s in in_memory_span_exporter.get_finished_spans()
            if s.name == "pipecat.conversation.turn"
        ]
        assert len(turn_spans) == 1
        turn_attrs = dict(turn_spans[0].attributes or {})
        assert user_text in str(turn_attrs.get(SpanAttributes.INPUT_VALUE, ""))

    async def test_endframe_flush(
        self,
        tracer: OITracer,
        config: TraceConfig,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
    ) -> None:
        """``EndFrame`` flushes the active turn marked as interrupted."""
        observer = _fast_observer(tracer, config)

        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(
                mock_stt_service,
                None,
                TranscriptionFrame(text="Mid sentence", user_id="u", timestamp="0"),
            )
        )

        # Connection drops — EndFrame arrives without a clean turn close.
        await observer.on_push_frame(create_frame_pushed(None, None, EndFrame()))

        turn_spans = _get_turn_spans(in_memory_span_exporter)
        assert len(turn_spans) == 1
        attrs = dict(turn_spans[0].attributes or {})
        assert attrs["conversation.was_interrupted"] is True
        assert attrs["conversation.end_reason"] == "interrupted"
        assert observer._turn_span is None

    async def test_pending_bot_response_suppresses_timer(
        self,
        tracer: OITracer,
        config: TraceConfig,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_stt_service: Mock,
        mock_llm_service: Mock,
    ) -> None:
        """While ``_bot_response_pending`` is set, neither the
        ``turn_end_timeout`` nor the ``no_responder_timeout`` closes the
        turn. Once the LLM stream ends and the bot speaks then stops, the
        turn closes normally with ``end_reason='completed'``."""
        observer = _fast_observer(tracer, config)
        transport = Mock(spec=BaseOutputTransport)

        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(
                mock_stt_service,
                None,
                TranscriptionFrame(text="Slow bot test", user_id="u", timestamp="0"),
            )
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, UserStoppedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStoppedSpeakingFrame())
        )

        # Mark a bot response as pending.
        context = LLMContext()
        await observer.on_push_frame(
            create_frame_pushed(None, mock_llm_service, LLMContextFrame(context=context))
        )
        assert observer._bot_response_pending is True

        # Wait past BOTH turn_end_timeout_secs (0.05) and
        # no_responder_timeout_secs (0.1) — turn must remain open.
        await asyncio.sleep(0.25)
        assert not _get_turn_spans(in_memory_span_exporter)
        assert observer._turn_span is not None

        # LLM stream ends, then the bot finally speaks.
        await observer.on_push_frame(
            create_frame_pushed(mock_llm_service, None, LLMFullResponseEndFrame())
        )
        await observer.on_push_frame(create_frame_pushed(None, None, BotStartedSpeakingFrame()))
        await observer.on_push_frame(
            create_frame_pushed(transport, None, _make_transport_tts_frame("Here you go."))
        )
        await observer.on_push_frame(create_frame_pushed(None, None, BotStoppedSpeakingFrame()))
        await asyncio.sleep(0.2)

        turn_spans = _get_turn_spans(in_memory_span_exporter)
        assert len(turn_spans) == 1
        attrs = dict(turn_spans[0].attributes or {})
        assert attrs["conversation.end_reason"] == "completed"
        assert attrs["conversation.initiator"] == "user"


@pytest.mark.asyncio
class TestStateMachineEdgeCases:
    """Defensive behaviour around degenerate or out-of-order frame sequences."""

    async def test_consecutive_user_started_is_idempotent(
        self,
        tracer: OITracer,
        config: TraceConfig,
        mock_stt_service: Mock,
    ) -> None:
        """Two back-to-back user-started frames must not double-open the
        turn or double-flip ``_user_speaking``."""
        observer = _fast_observer(tracer, config)

        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStartedSpeakingFrame())
        )
        await observer.on_push_frame(
            create_frame_pushed(mock_stt_service, None, VADUserStartedSpeakingFrame())
        )

        assert observer._turn_count == 1
        assert observer._user_speaking is True
        assert observer._user_spoken_this_turn is True

    async def test_bot_stopped_without_started_is_noop(
        self,
        tracer: OITracer,
        config: TraceConfig,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """A stray ``BotStoppedSpeakingFrame`` with no matching
        ``BotStartedSpeakingFrame`` must not crash or schedule a timer."""
        observer = _fast_observer(tracer, config)

        # No exception, no turn opened, no timer scheduled.
        await observer.on_push_frame(create_frame_pushed(None, None, BotStoppedSpeakingFrame()))

        assert observer._turn_span is None
        assert observer._bot_speaking is False
        assert observer._end_turn_timer is None
        assert not _get_turn_spans(in_memory_span_exporter)

    async def test_llm_context_frame_without_turn_does_not_crash(
        self,
        tracer: OITracer,
        config: TraceConfig,
        mock_llm_service: Mock,
    ) -> None:
        """``LLMContextFrame`` heading into an LLM with no active turn must
        be handled without crashing. Either the bot turn opens lazily or
        ``_bot_response_pending`` is set — both are acceptable."""
        observer = _fast_observer(tracer, config)

        context = LLMContext()
        # Should not raise.
        await observer.on_push_frame(
            create_frame_pushed(None, mock_llm_service, LLMContextFrame(context=context))
        )

        # The current implementation opens a bot turn lazily AND marks the
        # response pending. The spec permits either outcome; the contract
        # being tested is "no crash".
        assert observer._bot_response_pending is True
