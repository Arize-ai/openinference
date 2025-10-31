"""Service type detection for Pipecat base classes."""

from typing import Optional


class _ServiceDetector:
    """Detect service types from Pipecat base classes."""

    def detect_service_type(self, processor) -> Optional[str]:
        """
        Detect if a processor is an LLM, TTS, or STT service.

        Args:
            processor: A Pipecat FrameProcessor instance

        Returns:
            "llm", "tts", "stt", or None if not a recognized service
        """
        try:
            from pipecat.services.ai_services import LLMService, STTService, TTSService

            # Check against base classes - works for ALL implementations
            if isinstance(processor, STTService):
                return "stt"
            elif isinstance(processor, LLMService):
                return "llm"
            elif isinstance(processor, TTSService):
                return "tts"
        except ImportError:
            pass

        return None

    def get_provider_from_service(self, service) -> str:
        """
        Extract provider name from module path.

        Args:
            service: A Pipecat service instance

        Returns:
            Provider name (e.g., "openai", "anthropic") or "unknown"

        Example:
            Module: "pipecat.services.openai.llm" -> "openai"
        """
        module = service.__class__.__module__
        parts = module.split(".")

        # Module format: pipecat.services.{provider}.{service_type}
        if len(parts) >= 3 and parts[0] == "pipecat" and parts[1] == "services":
            return parts[2]

        return "unknown"

    def extract_service_metadata(self, service) -> dict:
        """
        Extract basic metadata from service instance.

        Args:
            service: A Pipecat service instance

        Returns:
            Dictionary with metadata (provider, model, voice, etc.)
        """
        metadata = {}

        # Provider from module path
        metadata["provider"] = self.get_provider_from_service(service)

        # Common attributes across services
        if hasattr(service, "_model"):
            metadata["model"] = service._model

        # TTS-specific
        if hasattr(service, "_voice"):
            metadata["voice"] = service._voice

        if hasattr(service, "_voice_id"):
            metadata["voice_id"] = service._voice_id

        return metadata
