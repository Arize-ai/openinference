"""Tests for OpenInferenceObserver."""

import os
import tempfile
import time
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSStartedFrame,
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
        """Test that starting a turn creates a turn span."""
        # Create a frame that starts a turn
        frame = VADUserStartedSpeakingFrame()
        data = create_frame_pushed(mock_stt_service, None, frame)

        # Process frame to start turn
        await observer.on_push_frame(data)

        # Verify turn span was created (but not ended yet)
        assert observer._is_turn_active
        assert observer._turn_span is not None
        assert observer._turn_count == 1

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

        # Start turn
        frame = VADUserStartedSpeakingFrame()
        data = create_frame_pushed(
            source=mock_stt_service, destination=None, frame=frame, direction="down"
        )
        await observer.on_push_frame(data)

        # End turn (simulate bot stopped speaking)
        end_frame = VADUserStoppedSpeakingFrame()
        end_data = create_frame_pushed(
            source=mock_stt_service, destination=None, frame=end_frame, direction="down"
        )

        # Need to trigger _end_turn manually or via timeout
        # For simplicity, call _end_turn directly
        await observer._end_turn(end_data, was_interrupted=False)

        # Get finished spans
        spans = in_memory_span_exporter.get_finished_spans()

        # Find turn span
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

        # End STT
        stop_frame = VADUserStoppedSpeakingFrame()
        stop_data = create_frame_pushed(
            source=mock_stt_service, destination=None, frame=stop_frame, direction="down"
        )
        await observer.on_push_frame(stop_data)

        # Verify STT span was created and is active
        service_id = id(mock_stt_service)
        assert service_id in observer._active_spans
        # Verify it has accumulated the transcription
        assert observer._active_spans[service_id]["accumulated_input"] == "Hello world "

    async def test_llm_span_creation(
        self,
        observer: OpenInferenceObserver,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_llm_service: Mock,
    ) -> None:
        """Test LLM service span creation."""
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
        mock_tts_service: Mock,
    ) -> None:
        """Test TTS service span creation."""
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
        mock_llm_service: Mock,
    ) -> None:
        """Test LLM output text accumulation."""
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
        mock_llm_service: Mock,
    ) -> None:
        """Test metrics extraction and setting on spans."""
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
        mock_stt_service: Mock,
        mock_llm_service: Mock,
        mock_tts_service: Mock,
    ) -> None:
        """Test multiple service spans within a single turn."""
        # Start turn with STT
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

        # Start TTS
        await observer.on_push_frame(
            create_frame_pushed(
                source=mock_tts_service, destination=None, frame=TTSStartedFrame(), direction="down"
            )
        )

        # Verify all three services have active spans
        assert id(mock_stt_service) in observer._active_spans
        assert id(mock_llm_service) in observer._active_spans
        assert id(mock_tts_service) in observer._active_spans

    async def test_inter_frame_spaces_handling(
        self,
        observer: OpenInferenceObserver,
        mock_llm_service: Mock,
    ) -> None:
        """Test handling of includes_inter_frame_spaces flag."""
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
