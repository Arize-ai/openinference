"""Service type detection for Pipecat base classes."""

from typing import Any, Dict, Optional

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.ai_service import AIService
from pipecat.services.image_service import ImageGenService
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.services.vision_service import VisionService
from pipecat.services.websocket_service import WebsocketService


class _ServiceDetector:
    """Detect service types from Pipecat base classes."""

    def detect_service_type(self, processor: FrameProcessor) -> Optional[str]:
        """
        Detect if a processor is an LLM, TTS, or STT service.

        Args:
            processor: A Pipecat FrameProcessor instance

        Returns:
            "llm", "tts", "stt", or None if not a recognized service
        """
        try:
            # Check against base classes - works for ALL implementations
            if isinstance(processor, LLMService):
                return "llm"
            elif isinstance(processor, STTService):
                return "stt"
            elif isinstance(processor, TTSService):
                return "tts"
            elif isinstance(processor, ImageGenService):
                return "image_gen"
            elif isinstance(processor, VisionService):
                return "vision"
            elif isinstance(processor, WebsocketService):
                return "websocket"
            elif isinstance(processor, AIService):
                return "ai_service"
        except ImportError:
            pass

        return None

    def get_provider_from_service(self, service: FrameProcessor) -> str:
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

    def extract_service_metadata(self, service: FrameProcessor) -> Dict[str, Any]:
        """
        Extract metadata from service instance based on service type.

        Args:
            service: A Pipecat service instance

        Returns:
            Dictionary with metadata (provider, model, voice, etc.)
        """
        metadata: Dict[str, Any] = {}

        provider = self.get_provider_from_service(service)
        service_type = self.detect_service_type(service)
        # Provider from module path
        metadata["provider"] = provider
        metadata["service_type"] = service_type

        # Extract attributes based on service type
        if service_type == "llm" and isinstance(service, LLMService):
            # LLM-specific attributes
            metadata["model"] = service.model_name
        elif service_type == "tts" and isinstance(service, TTSService):
            # TTS-specific attributes
            metadata["model"] = service.model_name
            metadata["voice_id"] = service._voice_id
            metadata["voice"] = service._voice_id  # Also add as "voice" for compatibility
            metadata["sample_rate"] = service.sample_rate
        elif service_type == "stt" and isinstance(service, STTService):
            # STT-specific attributes
            metadata["model"] = service.model_name
            metadata["is_muted"] = service.is_muted
            metadata["user_id"] = service._user_id
            metadata["sample_rate"] = service.sample_rate
        elif service_type == "image_gen" and isinstance(service, ImageGenService):
            # Image generation-specific attributes
            metadata["model"] = service.model_name
        elif service_type == "vision" and isinstance(service, VisionService):
            # Vision-specific attributes
            metadata["model"] = service.model_name

        return metadata
