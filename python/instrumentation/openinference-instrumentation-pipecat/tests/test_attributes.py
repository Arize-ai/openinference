"""Tests for attribute extraction from Pipecat frames and services."""

from unittest.mock import Mock

from pipecat.frames.frames import (
    FunctionCallResultFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TextFrame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.processors.aggregators.llm_context import LLMContext

from openinference.instrumentation.pipecat._attributes import (
    detect_provider_from_service,
    detect_service_type,
    detect_service_type_from_class_string,
    extract_attributes_from_frame,
    extract_service_attributes,
    get_model_name,
    safe_extract,
)
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes


class TestServiceDetection:
    """Test service type and provider detection functions."""

    def test_detect_service_type_llm(self, mock_llm_service: Mock) -> None:
        """Test LLM service type detection."""
        assert detect_service_type(mock_llm_service) == "llm"

    def test_detect_service_type_stt(self, mock_stt_service: Mock) -> None:
        """Test STT service type detection."""
        assert detect_service_type(mock_stt_service) == "stt"

    def test_detect_service_type_tts(self, mock_tts_service: Mock) -> None:
        """Test TTS service type detection."""
        assert detect_service_type(mock_tts_service) == "tts"

    def test_detect_service_type_unknown(self) -> None:
        """Test unknown service type returns 'unknown'."""
        mock_service = Mock()
        mock_service.__class__.__name__ = "UnknownService"
        assert detect_service_type(mock_service) == "unknown"

    def test_detect_service_type_from_class_string_llm(self) -> None:
        """Test LLM service detection from class string."""
        assert detect_service_type_from_class_string("OpenAILLMService") == "llm"
        assert detect_service_type_from_class_string("AnthropicLLMService") == "llm"

    def test_detect_service_type_from_class_string_stt(self) -> None:
        """Test STT service detection from class string."""
        assert detect_service_type_from_class_string("DeepgramSTTService") == "stt"

    def test_detect_service_type_from_class_string_tts(self) -> None:
        """Test TTS service detection from class string."""
        assert detect_service_type_from_class_string("CartesiaTTSService") == "tts"

    def test_detect_service_type_from_class_string_unknown(self) -> None:
        """Test unknown service type from class string."""
        assert detect_service_type_from_class_string("RandomService") == "unknown"

    def test_detect_provider_from_service_openai(self) -> None:
        """Test provider detection from OpenAI service."""
        service = Mock()
        service.__class__.__module__ = "pipecat.services.openai"
        assert detect_provider_from_service(service) == "openai"

    def test_detect_provider_from_service_deepgram(self) -> None:
        """Test provider detection from Deepgram service."""
        service = Mock()
        service.__class__.__module__ = "pipecat.services.deepgram"
        assert detect_provider_from_service(service) == "deepgram"

    def test_detect_provider_from_service_cartesia(self) -> None:
        """Test provider detection from Cartesia service."""
        service = Mock()
        service.__class__.__module__ = "pipecat.services.cartesia"
        assert detect_provider_from_service(service) == "cartesia"

    def test_detect_provider_from_service_unknown(self) -> None:
        """Test unknown provider detection."""
        service = Mock()
        service.__class__.__module__ = "some.random.module"
        assert detect_provider_from_service(service) == "unknown"


class TestModelNameExtraction:
    """Test model name extraction from services."""

    def test_get_model_name_from_full_model_name(self) -> None:
        """Test model name extraction from _full_model_name attribute."""
        service = Mock()
        service._full_model_name = "gpt-4-turbo-2024-04-09"
        service.model_name = "gpt-4"
        service.model = "gpt-4"
        assert get_model_name(service) == "gpt-4-turbo-2024-04-09"

    def test_get_model_name_from_model_name(self) -> None:
        """Test model name extraction from model_name attribute."""
        service = Mock()
        del service._full_model_name
        service.model_name = "claude-3-opus"
        service.model = "claude-3"
        assert get_model_name(service) == "claude-3-opus"

    def test_get_model_name_from_model(self) -> None:
        """Test model name extraction from model attribute."""
        service = Mock()
        del service._full_model_name
        del service.model_name
        service.model = "sonic-english"
        assert get_model_name(service) == "sonic-english"

    def test_get_model_name_fallback_unknown(self) -> None:
        """Test model name fallback to 'unknown'."""
        service = Mock()
        del service._full_model_name
        del service.model_name
        del service.model
        assert get_model_name(service) == "unknown"


class TestServiceAttributeExtraction:
    """Test service attribute extraction for span creation."""

    def test_extract_llm_service_attributes(self, mock_llm_service: Mock) -> None:
        """Test LLM service attribute extraction."""
        attributes = extract_service_attributes(mock_llm_service)

        assert (
            attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND]
            == OpenInferenceSpanKindValues.LLM.value
        )
        assert attributes[SpanAttributes.LLM_MODEL_NAME] == "gpt-4"
        assert attributes[SpanAttributes.LLM_PROVIDER] == "openai"
        assert attributes["gen_ai.system"] == "openai"
        assert attributes["gen_ai.request.model"] == "gpt-4"
        assert attributes["gen_ai.operation.name"] == "chat"
        assert attributes["gen_ai.output.type"] == "text"
        assert SpanAttributes.METADATA in attributes

    def test_extract_stt_service_attributes(self, mock_stt_service: Mock) -> None:
        """Test STT service attribute extraction."""
        attributes = extract_service_attributes(mock_stt_service)

        assert (
            attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND]
            == OpenInferenceSpanKindValues.LLM.value
        )
        assert attributes[SpanAttributes.LLM_MODEL_NAME] == "nova-2"
        assert attributes[SpanAttributes.LLM_PROVIDER] == "deepgram"
        assert attributes["service.model"] == "nova-2"
        assert attributes["audio.sample_rate"] == 16000

    def test_extract_tts_service_attributes(self, mock_tts_service: Mock) -> None:
        """Test TTS service attribute extraction."""
        attributes = extract_service_attributes(mock_tts_service)

        assert (
            attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND]
            == OpenInferenceSpanKindValues.LLM.value
        )
        assert attributes[SpanAttributes.LLM_MODEL_NAME] == "sonic-english"
        assert attributes[SpanAttributes.LLM_PROVIDER] == "cartesia"
        assert attributes["service.model"] == "sonic-english"
        assert attributes["audio.voice_id"] == "female-1"
        assert attributes["audio.voice"] == "female-1"
        assert attributes["audio.sample_rate"] == 22050


class TestFrameAttributeExtraction:
    """Test attribute extraction from different frame types."""

    def test_extract_transcription_frame_attributes(self) -> None:
        """Test transcription frame attribute extraction."""
        frame = TranscriptionFrame(text="Hello world", user_id="user123", timestamp=12345)
        attributes = extract_attributes_from_frame(frame)

        assert attributes[SpanAttributes.INPUT_VALUE] == "Hello world"
        assert attributes["llm.input_messages.0.message.role"] == "user"
        assert attributes["llm.input_messages.0.message.content"] == "Hello world"
        assert attributes["transcription.is_final"] is True
        assert attributes[SpanAttributes.USER_ID] == "user123"
        assert attributes["frame.timestamp"] == 12345

    def test_extract_llm_text_frame_attributes(self) -> None:
        """Test LLM text frame attribute extraction."""
        frame = LLMTextFrame(text="AI response chunk")
        attributes = extract_attributes_from_frame(frame)

        assert attributes["text.chunk"] == "AI response chunk"
        # OUTPUT_VALUE should NOT be set here - observer accumulates chunks
        assert SpanAttributes.OUTPUT_VALUE not in attributes

    def test_extract_tts_text_frame_attributes(self) -> None:
        """Test TTS text frame attribute extraction."""
        frame = TTSTextFrame(text="Speech text", aggregated_by="test")
        attributes = extract_attributes_from_frame(frame)

        assert attributes["text"] == "Speech text"
        assert attributes["text.chunk"] == "Speech text"

    def test_extract_text_frame_attributes(self) -> None:
        """Test generic text frame attribute extraction."""
        frame = TextFrame(text="Generic text")
        attributes = extract_attributes_from_frame(frame)

        assert attributes[SpanAttributes.OUTPUT_VALUE] == "Generic text"

    def test_extract_llm_context_frame_attributes(self) -> None:
        """Test LLM context frame attribute extraction."""
        # Create LLM context with messages
        context = LLMContext()
        context._messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is 2+2?"},
        ]

        frame = LLMContextFrame(context=context)
        attributes = extract_attributes_from_frame(frame)

        assert attributes["llm.messages_count"] == 2
        assert attributes["llm.input_messages.0.message.role"] == "system"
        assert attributes["llm.input_messages.0.message.content"] == "You are a helpful assistant"
        assert attributes["llm.input_messages.1.message.role"] == "user"
        assert attributes["llm.input_messages.1.message.content"] == "What is 2+2?"
        # input.value should be the LAST user message
        assert attributes[SpanAttributes.INPUT_VALUE] == "What is 2+2?"

    def test_extract_llm_context_frame_with_tools(self) -> None:
        """Test LLM context frame with tools attribute extraction."""
        context = LLMContext()
        context._messages = [{"role": "user", "content": "Use the weather tool"}]
        context._tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }
        ]

        frame = LLMContextFrame(context=context)
        attributes = extract_attributes_from_frame(frame)

        assert attributes["llm.tools_count"] == 1
        assert attributes["tools.names"] == "get_weather"
        assert "tools.definitions" in attributes

    def test_extract_llm_full_response_start_frame(self) -> None:
        """Test LLM full response start frame extraction."""
        frame = LLMFullResponseStartFrame()
        attributes = extract_attributes_from_frame(frame)

        assert attributes["llm.response_phase"] == "start"

    def test_extract_llm_full_response_end_frame(self) -> None:
        """Test LLM full response end frame extraction."""
        frame = LLMFullResponseEndFrame()
        attributes = extract_attributes_from_frame(frame)

        assert attributes["llm.response_phase"] == "end"

    def test_extract_function_call_result_frame(self) -> None:
        """Test function call result frame extraction."""
        frame = FunctionCallResultFrame(
            function_name="get_weather",
            tool_call_id="call_123",
            arguments={"location": "SF"},
            result={"temp": 72, "condition": "sunny"},
        )
        attributes = extract_attributes_from_frame(frame)

        assert attributes[SpanAttributes.TOOL_NAME] == "get_weather"
        assert "tool.result" in attributes


class TestMetricsExtraction:
    """Test metrics data extraction from metrics frames."""

    def test_extract_llm_usage_metrics(self) -> None:
        """Test LLM usage metrics extraction."""
        token_usage = LLMTokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cache_read_input_tokens=20,
        )
        metrics_data = LLMUsageMetricsData(
            processor="OpenAILLMService",
            model="gpt-4",
            value=token_usage,
        )
        frame = MetricsFrame(data=[metrics_data])
        attributes = extract_attributes_from_frame(frame)

        assert attributes["metrics.processor"] == "OpenAILLMService"
        assert attributes["metrics.model"] == "gpt-4"
        assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 100
        assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 50
        assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 150
        assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] == 20

    def test_extract_tts_usage_metrics(self) -> None:
        """Test TTS usage metrics extraction."""
        metrics_data = TTSUsageMetricsData(
            processor="CartesiaTTSService",
            model="sonic-english",
            value=42,  # character count
        )
        frame = MetricsFrame(data=[metrics_data])
        attributes = extract_attributes_from_frame(frame)

        assert attributes["metrics.processor"] == "CartesiaTTSService"
        assert attributes["tts.character_count"] == 42

    def test_extract_ttfb_metrics(self) -> None:
        """Test TTFB metrics extraction."""
        metrics_data = TTFBMetricsData(
            processor="OpenAILLMService",
            model="gpt-4",
            value=0.523,  # seconds
        )
        frame = MetricsFrame(data=[metrics_data])
        attributes = extract_attributes_from_frame(frame)

        assert attributes["service.ttfb_seconds"] == 0.523

    def test_extract_processing_metrics(self) -> None:
        """Test processing time metrics extraction."""
        metrics_data = ProcessingMetricsData(
            processor="DeepgramSTTService",
            model="nova-2",
            value=1.234,  # seconds
        )
        frame = MetricsFrame(data=[metrics_data])
        attributes = extract_attributes_from_frame(frame)

        assert attributes["service.processing_time_seconds"] == 1.234


class TestSafeExtraction:
    """Test safe extraction utility function."""

    def test_safe_extract_success(self) -> None:
        """Test successful extraction."""
        result = safe_extract(lambda: 42)
        assert result == 42

    def test_safe_extract_failure_returns_default(self) -> None:
        """Test extraction failure returns default value."""

        def failing_extractor() -> None:
            raise ValueError("Extraction failed")

        result = safe_extract(failing_extractor, default="fallback")
        assert result == "fallback"

    def test_safe_extract_failure_returns_none(self) -> None:
        """Test extraction failure returns None by default."""

        def failing_extractor() -> None:
            raise AttributeError("Missing attribute")

        result = safe_extract(failing_extractor)
        assert result is None
