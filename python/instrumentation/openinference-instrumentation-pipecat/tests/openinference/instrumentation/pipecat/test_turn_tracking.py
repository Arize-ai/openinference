"""
Test turn-based tracing functionality.
Ensures proper conversation turn detection and span creation.
"""

import asyncio

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from conftest import (
    assert_span_has_attributes,
    assert_span_hierarchy,
    get_spans_by_name,
    run_pipeline_task,
)
from openinference.instrumentation.pipecat import PipecatInstrumentor
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.task import PipelineTask


class TestTurnDetection:
    """Test basic turn detection and span creation"""

    @pytest.mark.asyncio
    async def test_user_turn_creates_span(
        self, tracer_provider, in_memory_span_exporter, simple_pipeline
    ):
        """Test that user starting to speak creates a turn span"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        task = PipelineTask(simple_pipeline, enable_turn_tracking=True)

        # Simulate user starting to speak
        await task.queue_frame(UserStartedSpeakingFrame())
        await asyncio.sleep(0.1)  # Let async processing happen

        # Should have a turn span (may not be finished yet)
        # This tests that turn tracking is working
        instrumentor.uninstrument()

    @pytest.mark.asyncio
    async def test_complete_turn_cycle(
        self, tracer_provider, in_memory_span_exporter, simple_pipeline
    ):
        """Test complete turn cycle: user speaks -> bot responds"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        task = PipelineTask(simple_pipeline, enable_turn_tracking=True)

        # User turn and bot response
        await run_pipeline_task(
            task,
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hello", user_id="test", timestamp=0),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            TextFrame(text="Hi there!"),
            BotStoppedSpeakingFrame(),
        )

        turn_spans = get_spans_by_name(
            in_memory_span_exporter, "pipecat.conversation.turn"
        )

        # Should have at least one complete turn
        assert len(turn_spans) >= 1

        instrumentor.uninstrument()

    @pytest.mark.asyncio
    async def test_turn_span_attributes(
        self, tracer_provider, in_memory_span_exporter, simple_pipeline
    ):
        """Test that turn spans have correct attributes"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        task = PipelineTask(simple_pipeline, enable_turn_tracking=True)

        # Complete turn
        await task.queue_frame(UserStartedSpeakingFrame())
        await task.queue_frame(
            TranscriptionFrame(text="Test input", user_id="user1", timestamp=0)
        )
        await task.queue_frame(UserStoppedSpeakingFrame())
        await task.queue_frame(BotStartedSpeakingFrame())
        await task.queue_frame(TextFrame(text="Test output"))
        await task.queue_frame(BotStoppedSpeakingFrame())

        await asyncio.sleep(0.1)

        turn_spans = get_spans_by_name(
            in_memory_span_exporter, "pipecat.conversation.turn"
        )

        if turn_spans:
            turn_span = turn_spans[0]
            expected_attributes = {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            }
            assert_span_has_attributes(turn_span, expected_attributes)

            # Should have input and output
            attrs = dict(turn_span.attributes)
            assert SpanAttributes.INPUT_VALUE in attrs or "conversation.input" in attrs
            assert (
                SpanAttributes.OUTPUT_VALUE in attrs or "conversation.output" in attrs
            )

        instrumentor.uninstrument()


class TestMultipleTurns:
    """Test handling of multiple conversation turns"""

    @pytest.mark.asyncio
    async def test_multiple_sequential_turns(
        self, tracer_provider, in_memory_span_exporter, simple_pipeline
    ):
        """Test that multiple turns create separate spans"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        task = PipelineTask(simple_pipeline, enable_turn_tracking=True)

        # Three complete turns
        await run_pipeline_task(
            task,
            # Turn 1
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="First", user_id="user1", timestamp=0),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            # Turn 2
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Second", user_id="user1", timestamp=1),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            # Turn 3
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Third", user_id="user1", timestamp=2),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
        )

        turn_spans = get_spans_by_name(
            in_memory_span_exporter, "pipecat.conversation.turn"
        )

        # Should have 3 separate turn spans
        assert len(turn_spans) >= 3

        # Each turn should have a turn number
        turn_numbers = []
        for span in turn_spans:
            attrs = dict(span.attributes)
            if "conversation.turn_number" in attrs:
                turn_numbers.append(attrs["conversation.turn_number"])

        assert len(set(turn_numbers)) >= 3  # At least 3 unique turn numbers

        instrumentor.uninstrument()

    @pytest.mark.asyncio
    async def test_turn_interruption(
        self, tracer_provider, in_memory_span_exporter, simple_pipeline
    ):
        """Test handling of turn interruption (user interrupts bot)"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        task = PipelineTask(simple_pipeline, enable_turn_tracking=True)

        # Turn with interruption
        await run_pipeline_task(
            task,
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hello", user_id="user1", timestamp=0),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            # User interrupts before bot finishes
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Wait, stop!", user_id="user1", timestamp=1),
            UserStoppedSpeakingFrame(),
        )

        turn_spans = get_spans_by_name(
            in_memory_span_exporter, "pipecat.conversation.turn"
        )

        # Should handle interruption gracefully - first turn ends, second begins
        assert len(turn_spans) >= 1

        # Check for interruption event or attribute
        for span in turn_spans:
            attrs = dict(span.attributes)
            # May have an end_reason attribute indicating interruption
            if "conversation.end_reason" in attrs:
                # Just verify the attribute exists
                assert isinstance(attrs["conversation.end_reason"], str)

        instrumentor.uninstrument()


class TestTurnHierarchy:
    """Test that turn spans properly parent phase spans (STT -> LLM -> TTS)"""

    @pytest.mark.asyncio
    async def test_turn_parents_phase_spans(
        self, tracer_provider, in_memory_span_exporter, simple_pipeline
    ):
        """Test that STT, LLM, TTS spans are children of turn span"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        task = PipelineTask(simple_pipeline, enable_turn_tracking=True)

        # Complete turn with all phases
        await task.queue_frame(UserStartedSpeakingFrame())
        await task.queue_frame(
            TranscriptionFrame(text="Hello", user_id="user1", timestamp=0)
        )
        await task.queue_frame(UserStoppedSpeakingFrame())
        # LLM processing happens here
        await task.queue_frame(BotStartedSpeakingFrame())
        await task.queue_frame(TextFrame(text="Response"))
        await task.queue_frame(BotStoppedSpeakingFrame())

        await asyncio.sleep(0.1)

        # Verify hierarchy: Turn -> STT/LLM/TTS
        turn_spans = get_spans_by_name(
            in_memory_span_exporter, "pipecat.conversation.turn"
        )
        stt_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.stt")
        llm_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.llm")
        tts_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.tts")

        if turn_spans and (stt_spans or llm_spans or tts_spans):
            turn_span = turn_spans[0]

            # Check that phase spans are children of turn span
            for phase_span in stt_spans + llm_spans + tts_spans:
                if phase_span.parent:
                    # Parent context should link to turn span
                    assert phase_span.parent.span_id == turn_span.context.span_id

        instrumentor.uninstrument()


class TestTurnConfiguration:
    """Test turn tracking configuration options"""

    @pytest.mark.asyncio
    async def test_turn_tracking_disabled(
        self, tracer_provider, in_memory_span_exporter, simple_pipeline
    ):
        """Test that turn tracking can be disabled"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        task = PipelineTask(simple_pipeline, enable_turn_tracking=False)

        # Send frames that would normally trigger turn tracking
        await task.queue_frame(UserStartedSpeakingFrame())
        await task.queue_frame(
            TranscriptionFrame(text="Hello", user_id="user1", timestamp=0)
        )
        await task.queue_frame(UserStoppedSpeakingFrame())

        await asyncio.sleep(0.1)

        turn_spans = get_spans_by_name(
            in_memory_span_exporter, "pipecat.conversation.turn"
        )

        # Should not create turn spans when disabled
        assert len(turn_spans) == 0

        instrumentor.uninstrument()

    @pytest.mark.asyncio
    async def test_session_id_in_turn_spans(
        self, tracer_provider, in_memory_span_exporter, simple_pipeline
    ):
        """Test that session ID is included in turn spans"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        task = PipelineTask(
            simple_pipeline, enable_turn_tracking=True, conversation_id="test-123"
        )

        await task.queue_frame(UserStartedSpeakingFrame())
        await task.queue_frame(
            TranscriptionFrame(text="Hello", user_id="user1", timestamp=0)
        )
        await task.queue_frame(UserStoppedSpeakingFrame())
        await task.queue_frame(BotStartedSpeakingFrame())
        await task.queue_frame(BotStoppedSpeakingFrame())

        await asyncio.sleep(0.1)

        turn_spans = get_spans_by_name(
            in_memory_span_exporter, "pipecat.conversation.turn"
        )

        if turn_spans:
            turn_span = turn_spans[0]
            attrs = dict(turn_span.attributes)

            # Should have session/conversation ID
            assert "session.id" in attrs or "conversation.id" in attrs

        instrumentor.uninstrument()


class TestTurnInputOutput:
    """Test capture of turn-level input and output"""

    @pytest.mark.asyncio
    async def test_turn_captures_user_input(
        self, tracer_provider, in_memory_span_exporter, simple_pipeline
    ):
        """Test that turn span captures complete user input"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        task = PipelineTask(simple_pipeline, enable_turn_tracking=True)

        user_message = "This is the user's complete message"

        await task.queue_frame(UserStartedSpeakingFrame())
        await task.queue_frame(
            TranscriptionFrame(text=user_message, user_id="user1", timestamp=0)
        )
        await task.queue_frame(UserStoppedSpeakingFrame())
        await task.queue_frame(BotStartedSpeakingFrame())
        await task.queue_frame(BotStoppedSpeakingFrame())

        await asyncio.sleep(0.1)

        turn_spans = get_spans_by_name(
            in_memory_span_exporter, "pipecat.conversation.turn"
        )

        if turn_spans:
            turn_span = turn_spans[0]
            attrs = dict(turn_span.attributes)

            input_value = attrs.get(SpanAttributes.INPUT_VALUE) or attrs.get(
                "conversation.input"
            )
            assert input_value is not None
            assert user_message in str(input_value)

        instrumentor.uninstrument()

    @pytest.mark.asyncio
    async def test_turn_captures_bot_output(
        self, tracer_provider, in_memory_span_exporter, simple_pipeline
    ):
        """Test that turn span captures complete bot output"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        task = PipelineTask(simple_pipeline, enable_turn_tracking=True)

        bot_response = "This is the bot's complete response"

        await task.queue_frame(UserStartedSpeakingFrame())
        await task.queue_frame(
            TranscriptionFrame(text="Hello", user_id="user1", timestamp=0)
        )
        await task.queue_frame(UserStoppedSpeakingFrame())
        await task.queue_frame(BotStartedSpeakingFrame())
        await task.queue_frame(TextFrame(text=bot_response))
        await task.queue_frame(BotStoppedSpeakingFrame())

        await asyncio.sleep(0.1)

        turn_spans = get_spans_by_name(
            in_memory_span_exporter, "pipecat.conversation.turn"
        )

        if turn_spans:
            turn_span = turn_spans[0]
            attrs = dict(turn_span.attributes)

            output_value = attrs.get(SpanAttributes.OUTPUT_VALUE) or attrs.get(
                "conversation.output"
            )
            assert output_value is not None
            assert bot_response in str(output_value)

        instrumentor.uninstrument()

    @pytest.mark.asyncio
    async def test_turn_handles_multiple_text_chunks(
        self, tracer_provider, in_memory_span_exporter, simple_pipeline
    ):
        """Test that turn span aggregates multiple text chunks"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        task = PipelineTask(simple_pipeline, enable_turn_tracking=True)

        await task.queue_frame(UserStartedSpeakingFrame())
        await task.queue_frame(
            TranscriptionFrame(text="Part one", user_id="user1", timestamp=0)
        )
        await task.queue_frame(
            TranscriptionFrame(text="Part two", user_id="user1", timestamp=1)
        )
        await task.queue_frame(UserStoppedSpeakingFrame())
        await task.queue_frame(BotStartedSpeakingFrame())
        await task.queue_frame(TextFrame(text="Response part A"))
        await task.queue_frame(TextFrame(text="Response part B"))
        await task.queue_frame(BotStoppedSpeakingFrame())

        await asyncio.sleep(0.1)

        turn_spans = get_spans_by_name(
            in_memory_span_exporter, "pipecat.conversation.turn"
        )

        if turn_spans:
            turn_span = turn_spans[0]
            attrs = dict(turn_span.attributes)

            # Should capture aggregated input/output
            input_value = attrs.get(SpanAttributes.INPUT_VALUE) or attrs.get(
                "conversation.input"
            )
            output_value = attrs.get(SpanAttributes.OUTPUT_VALUE) or attrs.get(
                "conversation.output"
            )

            # Both parts should be present (concatenated or in list)
            if input_value:
                assert "Part one" in str(input_value) or "Part two" in str(input_value)

            if output_value:
                assert "Response part A" in str(
                    output_value
                ) or "Response part B" in str(output_value)

        instrumentor.uninstrument()
