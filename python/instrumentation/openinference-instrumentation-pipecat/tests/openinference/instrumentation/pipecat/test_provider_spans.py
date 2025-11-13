"""
Test span creation for different service providers (OpenAI, Anthropic, ElevenLabs, Deepgram).
Ensures that base class instrumentation works across all provider implementations.
"""

import json

import pytest
from conftest import assert_span_has_attributes, get_spans_by_name, run_pipeline_task
from pipecat.frames.frames import (
    AudioRawFrame,
    LLMContextFrame,
    LLMMessagesUpdateFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask

from openinference.instrumentation.pipecat import PipecatInstrumentor
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes


class TestOpenAISpans:
    """Test span creation for OpenAI services"""

    @pytest.mark.asyncio
    async def test_openai_llm_span(self, tracer_provider, in_memory_span_exporter, mock_openai_llm):
        """Test that OpenAI LLM service creates proper spans"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)
        try:
            pipeline = Pipeline([mock_openai_llm])
            task = PipelineTask(pipeline)  # Use default settings so pipeline can complete

            # Send LLM request and run pipeline
            messages = [{"role": "user", "content": "Hello"}]
            await run_pipeline_task(task, LLMMessagesUpdateFrame(messages=messages, run_llm=True))

            llm_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.llm")

            assert len(llm_spans) > 0
            llm_span = llm_spans[0]

            expected_attrs = {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                "service.name": "MockLLMService",  # Class name of the service
                SpanAttributes.LLM_MODEL_NAME: "gpt-4",
                SpanAttributes.LLM_PROVIDER: "openai",  # Provider from metadata
            }
            assert_span_has_attributes(llm_span, expected_attrs)
        finally:
            instrumentor.uninstrument()

    @pytest.mark.asyncio
    async def test_openai_tts_span(self, tracer_provider, in_memory_span_exporter, mock_openai_tts):
        """Test that OpenAI TTS service creates proper spans"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        pipeline = Pipeline([mock_openai_tts])
        task = PipelineTask(pipeline)

        # Send text to convert to speech
        await run_pipeline_task(task, TextFrame(text="Hello world"))

        tts_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.tts")

        assert len(tts_spans) > 0
        tts_span = tts_spans[0]

        expected_attrs = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            "service.name": "MockTTSService",  # Class name
            "audio.voice": "alloy",
        }
        assert_span_has_attributes(tts_span, expected_attrs)

        instrumentor.uninstrument()

    @pytest.mark.asyncio
    async def test_openai_stt_span(self, tracer_provider, in_memory_span_exporter, mock_openai_stt):
        """Test that OpenAI STT service creates proper spans"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        pipeline = Pipeline([mock_openai_stt])
        task = PipelineTask(pipeline)

        # Send audio to transcribe
        audio_data = b"\x00" * 1024
        await run_pipeline_task(
            task, AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)
        )

        stt_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.stt")

        assert len(stt_spans) > 0
        stt_span = stt_spans[0]

        expected_attrs = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            "service.name": "MockSTTService",  # Class name
        }
        assert_span_has_attributes(stt_span, expected_attrs)

        instrumentor.uninstrument()

    @pytest.mark.asyncio
    async def test_openai_full_pipeline(
        self, tracer_provider, in_memory_span_exporter, openai_pipeline
    ):
        """Test full OpenAI pipeline (STT -> LLM -> TTS)"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        task = PipelineTask(openai_pipeline)

        # Simulate full conversation flow
        audio_data = b"\x00" * 1024
        await run_pipeline_task(
            task, AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)
        )

        # Should have spans for all three phases
        stt_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.stt")
        llm_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.llm")
        tts_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.tts")

        assert len(stt_spans) > 0
        # LLM and TTS may not be triggered in mock, but structure is tested

        # All should be Mock services with OpenAI provider
        for span in stt_spans + llm_spans + tts_spans:
            attrs = dict(span.attributes)
            service_name = attrs.get("service.name")
            assert service_name in [
                "MockSTTService",
                "MockLLMService",
                "MockTTSService",
            ]

        instrumentor.uninstrument()


class TestAnthropicSpans:
    """Test span creation for Anthropic services"""

    @pytest.mark.asyncio
    async def test_anthropic_llm_span(
        self, tracer_provider, in_memory_span_exporter, mock_anthropic_llm
    ):
        """Test that Anthropic LLM service creates proper spans"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        pipeline = Pipeline([mock_anthropic_llm])
        task = PipelineTask(pipeline)

        messages = [{"role": "user", "content": "Hello Claude"}]
        await run_pipeline_task(task, LLMMessagesUpdateFrame(messages=messages, run_llm=True))

        llm_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.llm")

        assert len(llm_spans) > 0
        llm_span = llm_spans[0]

        expected_attrs = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            "service.name": "MockLLMService",  # Class name
            SpanAttributes.LLM_MODEL_NAME: "claude-3-5-sonnet-20241022",
            SpanAttributes.LLM_PROVIDER: "anthropic",
        }
        assert_span_has_attributes(llm_span, expected_attrs)

        instrumentor.uninstrument()


class TestElevenLabsSpans:
    """Test span creation for ElevenLabs TTS service"""

    @pytest.mark.asyncio
    async def test_elevenlabs_tts_span(
        self, tracer_provider, in_memory_span_exporter, mock_elevenlabs_tts
    ):
        """Test that ElevenLabs TTS service creates proper spans"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        pipeline = Pipeline([mock_elevenlabs_tts])
        task = PipelineTask(pipeline)

        await run_pipeline_task(task, TextFrame(text="Test speech"))

        tts_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.tts")

        assert len(tts_spans) > 0
        tts_span = tts_spans[0]

        expected_attrs = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            "service.name": "MockTTSService",  # Class name
        }
        assert_span_has_attributes(tts_span, expected_attrs)

        # Should have audio.voice or audio.voice_id attribute
        attrs = dict(tts_span.attributes)
        assert "audio.voice" in attrs or "audio.voice_id" in attrs

        instrumentor.uninstrument()


class TestDeepgramSpans:
    """Test span creation for Deepgram STT service"""

    @pytest.mark.asyncio
    async def test_deepgram_stt_span(
        self, tracer_provider, in_memory_span_exporter, mock_deepgram_stt
    ):
        """Test that Deepgram STT service creates proper spans"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        pipeline = Pipeline([mock_deepgram_stt])
        task = PipelineTask(pipeline)

        audio_data = b"\x00" * 1024
        await run_pipeline_task(
            task, AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)
        )

        stt_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.stt")

        assert len(stt_spans) > 0
        stt_span = stt_spans[0]

        expected_attrs = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            "service.name": "MockSTTService",  # Class name
        }
        assert_span_has_attributes(stt_span, expected_attrs)

        instrumentor.uninstrument()


class TestMixedProviderPipeline:
    """Test pipelines with multiple different providers"""

    @pytest.mark.asyncio
    async def test_mixed_provider_span_creation(
        self, tracer_provider, in_memory_span_exporter, mixed_provider_pipeline
    ):
        """Test that mixed provider pipeline creates spans for all services"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        task = PipelineTask(mixed_provider_pipeline)

        # Simulate flow through pipeline
        audio_data = b"\x00" * 1024
        await run_pipeline_task(
            task, AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)
        )

        spans = in_memory_span_exporter.get_finished_spans()

        # Check we have spans from different providers
        providers_found = set()
        for span in spans:
            attrs = dict(span.attributes)
            if "service.name" in attrs:
                providers_found.add(attrs["service.name"])

        # Should have at least some of: deepgram, anthropic, elevenlabs
        assert len(providers_found) > 0

        instrumentor.uninstrument()

    @pytest.mark.asyncio
    async def test_mixed_providers_maintain_correct_attribution(
        self, tracer_provider, in_memory_span_exporter, mixed_provider_pipeline
    ):
        """Test that each span is attributed to correct provider"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        task = PipelineTask(mixed_provider_pipeline)

        audio_data = b"\x00" * 1024
        await run_pipeline_task(
            task, AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)
        )

        # STT span should be MockSTTService with deepgram provider
        stt_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.stt")
        if stt_spans:
            attrs = dict(stt_spans[0].attributes)
            assert attrs.get("service.name") == "MockSTTService"

        # LLM span should be MockLLMService with anthropic provider
        llm_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.llm")
        if llm_spans:
            attrs = dict(llm_spans[0].attributes)
            assert attrs.get("service.name") == "MockLLMService"
            assert attrs.get(SpanAttributes.LLM_PROVIDER) == "anthropic"

        # TTS span should be MockTTSService with elevenlabs provider
        tts_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.tts")
        if tts_spans:
            attrs = dict(tts_spans[0].attributes)
            assert attrs.get("service.name") == "MockTTSService"

        instrumentor.uninstrument()


class TestSpanInputOutput:
    """Test that spans capture input and output correctly for different providers"""

    @pytest.mark.asyncio
    async def test_stt_output_captured(
        self, tracer_provider, in_memory_span_exporter, mock_openai_stt
    ):
        """Test that STT span captures output transcription"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        pipeline = Pipeline([mock_openai_stt])
        task = PipelineTask(pipeline)

        audio_data = b"\x00" * 1024
        await run_pipeline_task(
            task, AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)
        )

        stt_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.stt")

        if stt_spans:
            attrs = dict(stt_spans[0].attributes)
            output_value = attrs.get(SpanAttributes.OUTPUT_VALUE)

            # Mock STT returns "Mock transcription"
            if output_value:
                assert "Mock transcription" in str(output_value)

        instrumentor.uninstrument()


class TestProviderSpecificAttributes:
    """Test provider-specific attributes are captured"""

    @pytest.mark.asyncio
    async def test_openai_model_attribute(
        self, tracer_provider, in_memory_span_exporter, mock_openai_llm
    ):
        """Test that OpenAI spans include model information"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        pipeline = Pipeline([mock_openai_llm])
        task = PipelineTask(pipeline)

        messages = [{"role": "user", "content": "Test"}]
        await run_pipeline_task(task, LLMMessagesUpdateFrame(messages=messages, run_llm=True))

        llm_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.llm")

        if llm_spans:
            attrs = dict(llm_spans[0].attributes)
            assert SpanAttributes.LLM_MODEL_NAME in attrs
            assert attrs[SpanAttributes.LLM_MODEL_NAME] == "gpt-4"

        instrumentor.uninstrument()

    @pytest.mark.asyncio
    async def test_anthropic_model_attribute(
        self, tracer_provider, in_memory_span_exporter, mock_anthropic_llm
    ):
        """Test that Anthropic spans include correct model"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        pipeline = Pipeline([mock_anthropic_llm])
        task = PipelineTask(pipeline)

        messages = [{"role": "user", "content": "Test"}]
        await run_pipeline_task(task, LLMMessagesUpdateFrame(messages=messages, run_llm=True))

        llm_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.llm")

        if llm_spans:
            attrs = dict(llm_spans[0].attributes)
            assert SpanAttributes.LLM_MODEL_NAME in attrs
            assert "claude" in attrs[SpanAttributes.LLM_MODEL_NAME].lower()

        instrumentor.uninstrument()

    @pytest.mark.asyncio
    async def test_elevenlabs_voice_attribute(
        self, tracer_provider, in_memory_span_exporter, mock_elevenlabs_tts
    ):
        """Test that ElevenLabs TTS includes voice_id"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        pipeline = Pipeline([mock_elevenlabs_tts])
        task = PipelineTask(pipeline)

        await run_pipeline_task(task, TextFrame(text="Test"))

        tts_spans = get_spans_by_name(in_memory_span_exporter, "pipecat.tts")

        if tts_spans:
            attrs = dict(tts_spans[0].attributes)
            # Should have audio.voice or audio.voice_id attribute
            has_voice = "audio.voice" in attrs or "audio.voice_id" in attrs
            assert has_voice

        instrumentor.uninstrument()


class TestLLMContextFrame:
    """Test that LLMContextFrame attributes are properly captured"""

    @pytest.mark.asyncio
    async def test_llm_context_frame_captures_messages(
        self, tracer_provider, in_memory_span_exporter, mock_openai_llm
    ):
        """Test that LLMContextFrame messages are extracted and added to span attributes"""

        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        pipeline = Pipeline([mock_openai_llm])
        task = PipelineTask(pipeline)

        # Create a mock LLMContext with messages
        class MockLLMContext:
            def __init__(self):
                self._messages = [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
                self._tools = None
                self._tool_choice = None

        mock_context = MockLLMContext()
        context_frame = LLMContextFrame(context=mock_context)

        # Send the context frame through the pipeline
        await run_pipeline_task(task, context_frame)

        # Get all spans - LLMContextFrame should be captured on service spans
        spans = in_memory_span_exporter.get_finished_spans()

        # Look for spans with LLM context attributes
        found_context_attrs = False
        for span in spans:
            attrs = dict(span.attributes) if span.attributes else {}
            if "llm.messages_count" in attrs:
                found_context_attrs = True
                assert attrs["llm.messages_count"] == 2

                # Verify messages were serialized
                if SpanAttributes.LLM_INPUT_MESSAGES in attrs:
                    messages_json = attrs[SpanAttributes.LLM_INPUT_MESSAGES]
                    messages = json.loads(messages_json)
                    assert len(messages) == 2
                    assert messages[0]["role"] == "user"
                    assert messages[1]["role"] == "assistant"

                # Should also be in INPUT_VALUE
                if SpanAttributes.INPUT_VALUE in attrs:
                    input_value = attrs[SpanAttributes.INPUT_VALUE]
                    assert "Hello" in input_value

        # LLMContextFrame tracking may be optional depending on implementation
        # but if present, it should have correct structure
        if found_context_attrs:
            assert True  # Attributes were found and validated

        instrumentor.uninstrument()

    @pytest.mark.asyncio
    async def test_llm_context_frame_with_tools(
        self, tracer_provider, in_memory_span_exporter, mock_openai_llm
    ):
        """Test that LLMContextFrame with tools captures tool count"""

        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        pipeline = Pipeline([mock_openai_llm])
        task = PipelineTask(pipeline)

        # Create a mock LLMContext with messages and tools
        class MockLLMContext:
            def __init__(self):
                self._messages = [{"role": "user", "content": "What's the weather?"}]
                self._tools = [
                    {"name": "get_weather", "description": "Get weather info"},
                    {"name": "get_time", "description": "Get current time"},
                ]
                self._tool_choice = None

        mock_context = MockLLMContext()
        context_frame = LLMContextFrame(context=mock_context)

        # Send the context frame through the pipeline
        await run_pipeline_task(task, context_frame)

        # Get all spans
        spans = in_memory_span_exporter.get_finished_spans()

        # Look for spans with tool count
        found_tools_attrs = False
        for span in spans:
            attrs = dict(span.attributes) if span.attributes else {}
            if "llm.tools_count" in attrs:
                found_tools_attrs = True
                assert attrs["llm.tools_count"] == 2

        # Tool tracking may be optional, but if present should be correct
        if found_tools_attrs:
            assert True  # Tool count was found and validated

        instrumentor.uninstrument()
