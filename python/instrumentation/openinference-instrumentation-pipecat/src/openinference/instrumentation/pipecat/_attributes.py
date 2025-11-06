"""Attribute extraction from Pipecat frames."""

import base64
import logging
from typing import Any, Callable, Dict, List

from openinference.instrumentation.helpers import safe_json_dumps
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesFrame,
    MetricsFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.processors.aggregators.llm_context import (
    LLMSpecificMessage,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.ai_service import AIService
from pipecat.services.image_service import ImageGenService
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.services.vision_service import VisionService
from pipecat.services.websocket_service import WebsocketService

logger = logging.getLogger(__name__)

__all__ = [
    "extract_attributes_from_frame",
    "extract_service_attributes",
    "detect_service_type",
    "detect_provider_from_service",
]


def safe_extract(extractor: Callable[[], Any], default: Any = None) -> Any:
    """
    Safely execute an extractor function, returning default value on error.

    Args:
        extractor: Function to execute
        default: Default value to return on error

    Returns:
        Result of extractor or default value on error
    """
    try:
        return extractor()
    except Exception as e:
        logger.debug(f"Failed to extract attribute: {e}")
        return default


def detect_service_type(service: FrameProcessor) -> str:
    """Detect the type of service."""
    if isinstance(service, STTService):
        return "stt"
    elif isinstance(service, LLMService):
        return "llm"
    elif isinstance(service, TTSService):
        return "tts"
    elif isinstance(service, ImageGenService):
        return "image_gen"
    elif isinstance(service, VisionService):
        return "vision"
    elif isinstance(service, WebsocketService):
        return "websocket"
    elif isinstance(service, AIService):
        return "ai"
    else:
        return "unknown"


def detect_provider_from_service(service: FrameProcessor) -> str:
    """Detect the provider from a service."""
    try:
        module = service.__class__.__module__
        parts = module.split(".")

        # Module format: pipecat.services.{provider}.{service_type}
        if len(parts) >= 3 and parts[0] == "pipecat" and parts[1] == "services":
            return parts[2]
        else:
            return "unknown"
    except Exception as e:
        logger.warning(f"Failed to detect provider from service: {e}")
        return "unknown"


class FrameAttributeExtractor:
    """Extract attributes from Pipecat frames."""

    attributes: Dict[str, Any] = {}

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for attribute, operation in self.attributes.items():
            # Use safe_extract to prevent individual attribute failures from breaking extraction
            value = safe_extract(lambda: operation(frame))
            if value is not None:
                result[attribute] = value
        return result


class TextFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from a text frame."""

    attributes: Dict[str, Any] = {
        "text.skip_tts": lambda frame: (frame.skip_tts if hasattr(frame, "skip_tts") else None),
    }

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        results = super().extract_from_frame(frame)
        if hasattr(frame, "text"):
            text = frame.text
            if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
                results[SpanAttributes.OUTPUT_VALUE] = text
            elif isinstance(frame, TextFrame):
                results[SpanAttributes.INPUT_VALUE] = text
            else:
                results[SpanAttributes.INPUT_VALUE] = text
        return results


# Singleton text frame extractor
_text_frame_extractor = TextFrameExtractor()


class AudioFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from an audio frame."""

    attributes: Dict[str, Any] = {
        "audio.wav": lambda frame: (
            base64.b64encode(frame.audio).decode("utf-8")
            if hasattr(frame, "audio") and frame.audio
            else None
        ),
        "audio.sample_rate": lambda frame: (getattr(frame, "sample_rate", None)),
        "audio.num_channels": lambda frame: (getattr(frame, "num_channels", None)),
        "audio.size_bytes": lambda frame: (len(getattr(frame, "audio", []))),
        "audio.frame_count": lambda frame: (getattr(frame, "num_frames", 0)),
    }


# Singleton audio frame extractor
_audio_frame_extractor = AudioFrameExtractor()


class LLMContextFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from an LLM context frame."""

    attributes: Dict[str, Any] = {
        "llm.messages_count": lambda frame: (
            len(frame.context._messages) if hasattr(frame.context, "_messages") else None
        ),
        "llm.messages": lambda frame: (
            safe_json_dumps(frame.context._messages)
            if hasattr(frame.context, "_messages")
            else None
        ),
    }


# Singleton LLM context frame extractor
_llm_context_frame_extractor = LLMContextFrameExtractor()


class LLMMessagesFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from an LLM messages frame."""

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        if hasattr(frame, "context") and frame.context:
            context = frame.context
            # Extract messages from context (context._messages is a list)
            if hasattr(context, "_messages") and context._messages:
                results["llm.messages_count"] = len(context._messages)

                # Convert messages to serializable format
                try:
                    # Messages can be LLMStandardMessage or LLMSpecificMessage
                    # They should be dict-like for serialization
                    messages_list: List[Any] = []
                    for msg in context._messages:
                        if isinstance(msg, dict):
                            raw_content = msg.content  # type: ignore
                            if isinstance(raw_content, str):
                                content = msg.content  # type: ignore
                            elif isinstance(raw_content, dict):
                                content = safe_json_dumps(raw_content)
                            else:
                                content = str(raw_content)
                            messages = {
                                "role": msg.role,  # type: ignore # LLMSpecificMessage does not have a role attribute
                                "content": content,
                                "name": msg.name if hasattr(msg, "name") else "",
                            }
                            messages_list.append(messages)
                        elif isinstance(msg, LLMSpecificMessage):
                            # Fallback: try to serialize the object
                            messages_list.append(msg.message)
                    messages_json = safe_json_dumps(messages_list)
                    results[SpanAttributes.LLM_INPUT_MESSAGES] = messages_json
                    results[SpanAttributes.INPUT_VALUE] = messages_json
                except (TypeError, ValueError, AttributeError) as e:
                    logger.debug(f"Could not serialize LLMContext messages: {e}")

            # Extract tools if present
            if hasattr(context, "_tools") and context._tools:
                try:
                    # Try to get tool count
                    if isinstance(context._tools, list):
                        results["llm.tools_count"] = len(context._tools)
                except (TypeError, AttributeError):
                    pass

        return results


# Singleton LLM messages frame extractor
_llm_messages_frame_extractor = LLMMessagesFrameExtractor()


class LLMMessagesSequenceFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from an LLM messages append frame."""

    phase: str = "append"

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "llm.response_phase": self.phase,
        }
        if hasattr(frame, "messages") and frame.messages:
            messages = frame.messages
            results["llm.messages_count"] = len(messages)

            # Extract text content for input.value
            user_messages = safe_json_dumps(messages)
            if user_messages:
                results[SpanAttributes.LLM_INPUT_MESSAGES] = user_messages
                results[SpanAttributes.INPUT_VALUE] = user_messages
        return results


# Singleton LLM messages sequence frame extractor
_llm_messages_sequence_frame_extractor = LLMMessagesSequenceFrameExtractor()


class LLMMessagesAppendFrameExtractor(LLMMessagesSequenceFrameExtractor):
    """Extract attributes from an LLM messages append frame."""

    phase: str = "append"


# Singleton LLM messages append frame extractor
_llm_messages_append_frame_extractor = LLMMessagesAppendFrameExtractor()


class LLMFullResponseStartFrameExtractor(LLMMessagesSequenceFrameExtractor):
    """Extract attributes from an LLM full response start frame."""

    phase: str = "start"


# Singleton LLM full response start frame extractor
_llm_full_response_start_frame_extractor = LLMFullResponseStartFrameExtractor()


class LLMFullResponseEndFrameExtractor(LLMMessagesSequenceFrameExtractor):
    """Extract attributes from an LLM full response end frame."""

    phase: str = "end"


# Singleton LLM full response end frame extractor
_llm_full_response_end_frame_extractor = LLMFullResponseEndFrameExtractor()


class FunctionCallFromLLMFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from function call frames."""

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        if hasattr(frame, "function_name") and frame.function_name:
            results[SpanAttributes.TOOL_NAME] = frame.function_name
        if hasattr(frame, "arguments") and frame.arguments:
            # Arguments are typically a dict
            if isinstance(frame.arguments, dict):
                params = safe_json_dumps(frame.arguments)
                if params:
                    results[SpanAttributes.TOOL_PARAMETERS] = params
            else:
                results[SpanAttributes.TOOL_PARAMETERS] = safe_extract(lambda: str(frame.arguments))
        if hasattr(frame, "tool_call_id") and frame.tool_call_id:
            results["tool.call_id"] = frame.tool_call_id
        return results


# Singleton function call from LLM frame extractor
_function_call_from_llm_frame_extractor = FunctionCallFromLLMFrameExtractor()


class FunctionCallResultFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from function call result frames."""

    attributes: Dict[str, Any] = {
        SpanAttributes.TOOL_NAME: lambda frame: getattr(frame, "function_name", None),
        SpanAttributes.OUTPUT_VALUE: lambda frame: (
            safe_json_dumps(frame.result)
            if hasattr(frame, "result") and isinstance(frame.result, (dict, list))
            else str(frame.result)
            if hasattr(frame, "result")
            else None
        ),
        "tool.call_id": lambda frame: getattr(frame, "tool_call_id", None),
    }


# Singleton function call result frame extractor
_function_call_result_frame_extractor = FunctionCallResultFrameExtractor()


class FunctionCallInProgressFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from function call in-progress frames."""

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        if hasattr(frame, "function_name") and frame.function_name:
            results[SpanAttributes.TOOL_NAME] = frame.function_name
            results["tool.status"] = "in_progress"
        return results


# Singleton function call in-progress frame extractor
_function_call_in_progress_frame_extractor = FunctionCallInProgressFrameExtractor()


class LLMTokenMetricsDataExtractor(FrameAttributeExtractor):
    """Extract attributes from LLM token metrics data."""

    attributes: Dict[str, Any] = {
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: lambda frame: getattr(frame, "prompt_tokens", None),
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: lambda frame: getattr(
            frame, "completion_tokens", None
        ),
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: lambda frame: getattr(frame, "total_tokens", None),
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: lambda frame: getattr(
            frame, "cache_read_input_tokens", None
        ),
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO: lambda frame: getattr(
            frame, "audio_tokens", None
        ),
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING: lambda frame: getattr(
            frame, "reasoning_tokens", None
        ),
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO: lambda frame: getattr(
            frame, "audio_tokens", None
        ),
    }


# Singleton LLM token metrics data extractor
_llm_token_metrics_data_extractor = LLMTokenMetricsDataExtractor()


class LLMUsageMetricsDataExtractor(FrameAttributeExtractor):
    """Extract attributes from LLM usage metrics data."""

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        if hasattr(frame, "value") and frame.value:
            return _llm_token_metrics_data_extractor.extract_from_frame(frame.value)
        return {}


# Singleton LLM usage metrics data extractor
_llm_usage_metrics_data_extractor = LLMUsageMetricsDataExtractor()


class TTSUsageMetricsDataExtractor(FrameAttributeExtractor):
    """Extract attributes from TTS usage metrics data."""

    attributes: Dict[str, Any] = {
        "tts.character_count": lambda frame: getattr(frame, "value", None),
    }


# Singleton TTS usage metrics data extractor
_tts_usage_metrics_data_extractor = TTSUsageMetricsDataExtractor()


class TTFBMetricsDataExtractor(FrameAttributeExtractor):
    """Extract attributes from TTFB metrics data."""

    attributes: Dict[str, Any] = {
        "service.ttfb_seconds": lambda frame: getattr(frame, "value", None),
    }


# Singleton TTFB metrics data extractor
_ttfb_metrics_data_extractor = TTFBMetricsDataExtractor()


class ProcessingMetricsDataExtractor(FrameAttributeExtractor):
    """Extract attributes from processing metrics data."""

    attributes: Dict[str, Any] = {
        "service.processing_time_seconds": lambda frame: getattr(frame, "value", None),
    }


# Singleton processing metrics data extractor
_processing_metrics_data_extractor = ProcessingMetricsDataExtractor()


class MetricsFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from metrics frames."""

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        if not hasattr(frame, "data") or not frame.data:
            return results

        for metrics_data in frame.data:
            # Check the type of metrics_data and extract accordingly
            if isinstance(metrics_data, LLMUsageMetricsData):
                results.update(
                    _llm_usage_metrics_data_extractor.extract_from_frame(metrics_data)  # type: ignore
                )
            elif isinstance(metrics_data, TTSUsageMetricsData):
                results.update(
                    _tts_usage_metrics_data_extractor.extract_from_frame(metrics_data)  # type: ignore
                )
            elif isinstance(metrics_data, TTFBMetricsData):
                results.update(
                    _ttfb_metrics_data_extractor.extract_from_frame(metrics_data)  # type: ignore
                )
            elif isinstance(metrics_data, ProcessingMetricsData):
                results.update(
                    _processing_metrics_data_extractor.extract_from_frame(metrics_data)  # type: ignore
                )

        return results


# Singleton metrics frame extractor
_metrics_frame_extractor = MetricsFrameExtractor()


class GenericFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from a generic frame."""

    attributes: Dict[str, Any] = {
        "frame.type": lambda frame: frame.__class__.__name__,
        "frame.id": lambda frame: frame.id,
        SpanAttributes.USER_ID: lambda frame: getattr(frame, "user_id", None),
        "frame.name": lambda frame: getattr(frame, "name", None),
        "frame.pts": lambda frame: getattr(frame, "pts", None),
        "frame.timestamp": lambda frame: getattr(frame, "timestamp", None),
        "frame.metadata": lambda frame: safe_json_dumps(getattr(frame, "metadata", {})),
        "frame.transport_source": lambda frame: getattr(frame, "transport_source", None),
        "frame.transport_destination": lambda frame: getattr(frame, "transport_destination", None),
        "frame.error.message": lambda frame: getattr(frame, "error", None),
    }

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        results = super().extract_from_frame(frame)

        # Use singleton instances to avoid creating new objects for every frame
        if isinstance(frame, TextFrame):
            results.update(_text_frame_extractor.extract_from_frame(frame))
        if isinstance(frame, AudioRawFrame):
            results.update(_audio_frame_extractor.extract_from_frame(frame))
        if isinstance(frame, LLMContextFrame):
            results.update(_llm_context_frame_extractor.extract_from_frame(frame))
        if isinstance(frame, LLMMessagesFrame):
            results.update(_llm_messages_frame_extractor.extract_from_frame(frame))
        if isinstance(frame, LLMMessagesAppendFrame):
            results.update(_llm_messages_append_frame_extractor.extract_from_frame(frame))
        if isinstance(frame, LLMFullResponseStartFrame):
            results.update(_llm_full_response_start_frame_extractor.extract_from_frame(frame))
        if isinstance(frame, LLMFullResponseEndFrame):
            results.update(_llm_full_response_end_frame_extractor.extract_from_frame(frame))
        if isinstance(frame, FunctionCallFromLLM):
            results.update(_function_call_from_llm_frame_extractor.extract_from_frame(frame))
        if isinstance(frame, FunctionCallResultFrame):
            results.update(_function_call_result_frame_extractor.extract_from_frame(frame))
        if isinstance(frame, FunctionCallInProgressFrame):
            results.update(_function_call_in_progress_frame_extractor.extract_from_frame(frame))
        if isinstance(frame, MetricsFrame):
            results.update(_metrics_frame_extractor.extract_from_frame(frame))
        return results


# Singleton generic frame extractor
_generic_frame_extractor = GenericFrameExtractor()


def extract_attributes_from_frame(frame: Frame) -> Dict[str, Any]:
    """
    Extract attributes from a frame using the singleton extractor.

    This is the main entry point for attribute extraction.
    """
    return _generic_frame_extractor.extract_from_frame(frame)


# ============================================================================
# Service Attribute Extraction (for span creation)
# ============================================================================


class ServiceAttributeExtractor:
    """Base class for extracting attributes from services for span creation."""

    attributes: Dict[str, Any] = {}

    def extract_from_service(self, service: FrameProcessor) -> Dict[str, Any]:
        """Extract attributes from a service."""
        result: Dict[str, Any] = {}
        for attribute, operation in self.attributes.items():
            # Use safe_extract to prevent individual attribute failures from breaking extraction
            value = safe_extract(lambda: operation(service))
            if value is not None:
                result[attribute] = value
        return result


class BaseServiceAttributeExtractor(ServiceAttributeExtractor):
    """Extract base attributes common to all services."""

    attributes: Dict[str, Any] = {
        "service.type": lambda service: detect_service_type(service),
        "service.provider": lambda service: detect_provider_from_service(service),
    }


# Singleton base service attribute extractor
_base_service_attribute_extractor = BaseServiceAttributeExtractor()


class LLMServiceAttributeExtractor(ServiceAttributeExtractor):
    """Extract attributes from an LLM service for span creation."""

    attributes: Dict[str, Any] = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: lambda service: (
            OpenInferenceSpanKindValues.LLM.value
        ),
        SpanAttributes.LLM_MODEL_NAME: lambda service: getattr(service, "model_name", None)
        or getattr(service, "model", None),
        SpanAttributes.LLM_PROVIDER: lambda service: detect_provider_from_service(service),
        "service.model": lambda service: getattr(service, "model_name", None)
        or getattr(service, "model", None),
    }

    def extract_from_service(self, service: FrameProcessor) -> Dict[str, Any]:
        """Extract LLM service attributes including settings."""
        results = super().extract_from_service(service)

        # Extract LLM settings/configuration as metadata
        if hasattr(service, "_settings"):
            if isinstance(service._settings, dict):
                results[SpanAttributes.METADATA] = safe_json_dumps(service._settings)
            else:
                results[SpanAttributes.METADATA] = str(service._settings)

        return results


# Singleton LLM service attribute extractor
_llm_service_attribute_extractor = LLMServiceAttributeExtractor()


class STTServiceAttributeExtractor(ServiceAttributeExtractor):
    """Extract attributes from an STT service for span creation."""

    attributes: Dict[str, Any] = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: lambda service: (
            OpenInferenceSpanKindValues.CHAIN.value
        ),
        SpanAttributes.LLM_MODEL_NAME: lambda service: getattr(service, "model_name", None)
        or getattr(service, "model", None),
        SpanAttributes.LLM_PROVIDER: lambda service: detect_provider_from_service(service),
        "service.model": lambda service: getattr(service, "model_name", None)
        or getattr(service, "model", None),
        "audio.sample_rate": lambda service: getattr(service, "sample_rate", None),
        "audio.is_muted": lambda service: getattr(service, "is_muted", None),
        "audio.user_id": lambda service: getattr(service, "_user_id", None),
    }


# Singleton STT service attribute extractor
_stt_service_attribute_extractor = STTServiceAttributeExtractor()


class TTSServiceAttributeExtractor(ServiceAttributeExtractor):
    """Extract attributes from a TTS service for span creation."""

    attributes: Dict[str, Any] = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: lambda service: (
            OpenInferenceSpanKindValues.CHAIN.value
        ),
        SpanAttributes.LLM_MODEL_NAME: lambda service: getattr(service, "model_name", None)
        or getattr(service, "model", None),
        SpanAttributes.LLM_PROVIDER: lambda service: detect_provider_from_service(service),
        "service.model": lambda service: getattr(service, "model_name", None)
        or getattr(service, "model", None),
        "audio.voice_id": lambda service: getattr(service, "_voice_id", None),
        "audio.voice": lambda service: getattr(service, "_voice_id", None),
        "audio.sample_rate": lambda service: getattr(service, "sample_rate", None),
    }


# Singleton TTS service attribute extractor
_tts_service_attribute_extractor = TTSServiceAttributeExtractor()


class ImageGenServiceAttributeExtractor(ServiceAttributeExtractor):
    """Extract attributes from an image generation service for span creation."""

    attributes: Dict[str, Any] = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: lambda service: (
            OpenInferenceSpanKindValues.CHAIN.value
        ),
        "service.model": lambda service: getattr(service, "model_name", None)
        or getattr(service, "model", None),
    }


# Singleton image gen service attribute extractor
_image_gen_service_attribute_extractor = ImageGenServiceAttributeExtractor()


class VisionServiceAttributeExtractor(ServiceAttributeExtractor):
    """Extract attributes from a vision service for span creation."""

    attributes: Dict[str, Any] = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: lambda service: (
            OpenInferenceSpanKindValues.CHAIN.value
        ),
        "service.model": lambda service: getattr(service, "model_name", None)
        or getattr(service, "model", None),
    }


# Singleton vision service attribute extractor
_vision_service_attribute_extractor = VisionServiceAttributeExtractor()


class WebsocketServiceAttributeExtractor(ServiceAttributeExtractor):
    """Extract attributes from a websocket service for span creation."""

    attributes: Dict[str, Any] = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: lambda service: (
            OpenInferenceSpanKindValues.CHAIN.value
        ),
        "websocket.reconnect_on_error": lambda service: getattr(
            service, "_reconnect_on_error", None
        ),
    }


# Singleton websocket service attribute extractor
_websocket_service_attribute_extractor = WebsocketServiceAttributeExtractor()


def extract_service_attributes(service: FrameProcessor) -> Dict[str, Any]:
    """
    Extract attributes from a service for span creation.

    This function is used when creating service spans to collect the right attributes
    based on the service type. It applies service-specific extractors to gather
    attributes like span kind, model name, provider, and service-specific configuration.

    Args:
        service: The service instance (FrameProcessor)

    Returns:
        Dictionary of attributes to set on the span
    """
    attributes: Dict[str, Any] = {}

    # Always extract base service attributes
    attributes.update(_base_service_attribute_extractor.extract_from_service(service))

    # Extract service-specific attributes based on type
    if isinstance(service, LLMService):
        attributes.update(_llm_service_attribute_extractor.extract_from_service(service))
    elif isinstance(service, STTService):
        attributes.update(_stt_service_attribute_extractor.extract_from_service(service))
    elif isinstance(service, TTSService):
        attributes.update(_tts_service_attribute_extractor.extract_from_service(service))
    elif isinstance(service, ImageGenService):
        attributes.update(_image_gen_service_attribute_extractor.extract_from_service(service))
    elif isinstance(service, VisionService):
        attributes.update(_vision_service_attribute_extractor.extract_from_service(service))
    elif isinstance(service, WebsocketService):
        attributes.update(_websocket_service_attribute_extractor.extract_from_service(service))

    return attributes
