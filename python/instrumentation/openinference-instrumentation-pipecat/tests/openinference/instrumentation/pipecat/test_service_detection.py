"""
Test service type detection and provider identification across different implementations.
This ensures our base class instrumentation affects all inheriting classes.
"""


class TestServiceTypeDetection:
    """Test detection of service types (LLM, TTS, STT) from base classes"""

    def test_detect_llm_service_base(self, mock_llm_service):
        """Test detection of generic LLM service"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        service_type = detector.detect_service_type(mock_llm_service)

        assert service_type == "llm"

    def test_detect_tts_service_base(self, mock_tts_service):
        """Test detection of generic TTS service"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        service_type = detector.detect_service_type(mock_tts_service)

        assert service_type == "tts"

    def test_detect_stt_service_base(self, mock_stt_service):
        """Test detection of generic STT service"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        service_type = detector.detect_service_type(mock_stt_service)

        assert service_type == "stt"

    def test_detect_openai_llm(self, mock_openai_llm):
        """Test detection of OpenAI LLM service"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        service_type = detector.detect_service_type(mock_openai_llm)

        assert service_type == "llm"

    def test_detect_anthropic_llm(self, mock_anthropic_llm):
        """Test detection of Anthropic LLM service"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        service_type = detector.detect_service_type(mock_anthropic_llm)

        assert service_type == "llm"

    def test_detect_elevenlabs_tts(self, mock_elevenlabs_tts):
        """Test detection of ElevenLabs TTS service"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        service_type = detector.detect_service_type(mock_elevenlabs_tts)

        assert service_type == "tts"

    def test_detect_deepgram_stt(self, mock_deepgram_stt):
        """Test detection of Deepgram STT service"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        service_type = detector.detect_service_type(mock_deepgram_stt)

        assert service_type == "stt"

    def test_detect_non_service_processor(self):
        """Test that non-service processors return None"""
        from pipecat.processors.frame_processor import FrameProcessor

        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        generic_processor = FrameProcessor()
        service_type = detector.detect_service_type(generic_processor)

        assert service_type is None


class TestProviderDetection:
    """Test provider detection from service module paths"""

    def test_openai_provider_detection(self, mock_openai_llm):
        """Test OpenAI provider detection from module path"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        provider = detector.get_provider_from_service(mock_openai_llm)

        assert provider == "openai"

    def test_anthropic_provider_detection(self, mock_anthropic_llm):
        """Test Anthropic provider detection"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        provider = detector.get_provider_from_service(mock_anthropic_llm)

        assert provider == "anthropic"

    def test_elevenlabs_provider_detection(self, mock_elevenlabs_tts):
        """Test ElevenLabs provider detection"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        provider = detector.get_provider_from_service(mock_elevenlabs_tts)

        assert provider == "elevenlabs"

    def test_deepgram_provider_detection(self, mock_deepgram_stt):
        """Test Deepgram provider detection"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        provider = detector.get_provider_from_service(mock_deepgram_stt)

        assert provider == "deepgram"

    def test_unknown_provider_fallback(self, mock_llm_service):
        """Test fallback for services without clear provider"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        provider = detector.get_provider_from_service(mock_llm_service)

        # Mock service has provider="mock" set explicitly
        assert provider in ["mock", "unknown"]


class TestServiceMetadataExtraction:
    """Test extraction of service metadata (model, voice, etc.)"""

    def test_extract_llm_model(self, mock_openai_llm):
        """Test extraction of LLM model name"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        metadata = detector.extract_service_metadata(mock_openai_llm)

        assert "model" in metadata
        assert metadata["model"] == "gpt-4"

    def test_extract_tts_model_and_voice(self, mock_openai_tts):
        """Test extraction of TTS model and voice"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        metadata = detector.extract_service_metadata(mock_openai_tts)

        assert "model" in metadata
        assert metadata["model"] == "tts-1"
        assert "voice" in metadata
        assert metadata["voice"] == "alloy"

    def test_extract_stt_model(self, mock_openai_stt):
        """Test extraction of STT model"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        metadata = detector.extract_service_metadata(mock_openai_stt)

        assert "model" in metadata
        assert metadata["model"] == "whisper-1"

    def test_extract_elevenlabs_voice_id(self, mock_elevenlabs_tts):
        """Test extraction of ElevenLabs voice_id"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        metadata = detector.extract_service_metadata(mock_elevenlabs_tts)

        assert "voice_id" in metadata or "voice" in metadata

    def test_extract_anthropic_model(self, mock_anthropic_llm):
        """Test extraction of Anthropic model"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        metadata = detector.extract_service_metadata(mock_anthropic_llm)

        assert "model" in metadata
        assert "claude" in metadata["model"].lower()

    def test_extract_provider_from_metadata(self, mock_openai_llm):
        """Test that provider is included in metadata"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        metadata = detector.extract_service_metadata(mock_openai_llm)

        assert "provider" in metadata
        assert metadata["provider"] == "openai"


class TestMultiProviderPipeline:
    """Test service detection in pipelines with multiple providers"""

    def test_detect_all_services_in_mixed_pipeline(self, mixed_provider_pipeline):
        """Test detection of all services in a pipeline with mixed providers"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        processors = mixed_provider_pipeline._processors

        service_types = [detector.detect_service_type(p) for p in processors]

        # Should detect STT, LLM, TTS in order
        assert "stt" in service_types
        assert "llm" in service_types
        assert "tts" in service_types

    def test_extract_providers_from_mixed_pipeline(self, mixed_provider_pipeline):
        """Test provider extraction from mixed provider pipeline"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        processors = mixed_provider_pipeline._processors

        providers = [detector.get_provider_from_service(p) for p in processors]

        # Should have deepgram, anthropic, elevenlabs
        assert "deepgram" in providers
        assert "anthropic" in providers
        assert "elevenlabs" in providers

    def test_extract_all_metadata_from_pipeline(self, mixed_provider_pipeline):
        """Test metadata extraction from all services in pipeline"""
        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        detector = _ServiceDetector()
        processors = mixed_provider_pipeline._processors

        metadata_list = [detector.extract_service_metadata(p) for p in processors]

        # Each should have metadata
        for metadata in metadata_list:
            assert "provider" in metadata
            # At least one should have a model
            if "model" in metadata:
                assert isinstance(metadata["model"], str)


class TestServiceInheritanceDetection:
    """Test that service detection works correctly with inheritance hierarchies"""

    def test_custom_llm_service_detected(self):
        """Test that custom LLM service inheriting from base is detected"""
        from pipecat.services.llm_service import LLMService

        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        class CustomLLMService(LLMService):
            def __init__(self):
                super().__init__()
                self._model = "custom-model"

        detector = _ServiceDetector()
        custom_service = CustomLLMService()
        service_type = detector.detect_service_type(custom_service)

        assert service_type == "llm"

    def test_deeply_nested_service_detected(self):
        """Test that services with deep inheritance are detected"""
        from pipecat.services.tts_service import TTSService

        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        class BaseTTSWrapper(TTSService):
            async def run_tts(self, text: str):
                yield

        class SpecificTTSService(BaseTTSWrapper):
            pass

        detector = _ServiceDetector()
        nested_service = SpecificTTSService()
        service_type = detector.detect_service_type(nested_service)

        assert service_type == "tts"

    def test_multiple_inheritance_service(self):
        """Test service detection with multiple inheritance (edge case)"""
        from pipecat.services.stt_service import STTService

        from openinference.instrumentation.pipecat._service_detector import (
            _ServiceDetector,
        )

        class MixinClass:
            pass

        class MultiInheritSTT(MixinClass, STTService):
            async def run_stt(self, audio: bytes):
                yield

        detector = _ServiceDetector()
        multi_service = MultiInheritSTT()
        service_type = detector.detect_service_type(multi_service)

        # Should still detect as STT since it inherits from STTService
        assert service_type == "stt"
