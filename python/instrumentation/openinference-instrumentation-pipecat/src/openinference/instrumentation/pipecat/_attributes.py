"""Attribute extraction from Pipecat frames."""

import logging
from typing import Any, Callable, Dict, List, Type

from openinference.instrumentation.helpers import safe_json_dumps
from openinference.semconv.trace import (
    AudioAttributes,
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)
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
    LLMTextFrame,
    MetricsFrame,
    TextFrame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    MetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
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
from pipecat.transports.base_output import BaseOutputTransport

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = [
    "extract_attributes_from_frame",
    "extract_service_attributes",
    "detect_service_type",
    "detect_service_type_from_class_string",
    "detect_provider_from_service",
]

FRAME_TYPE_MAP = {
    TranscriptionFrame.__name__: "transcription",
    TTSTextFrame.__name__: "tts_text",
    TextFrame.__name__: "text",
    AudioRawFrame.__name__: "audio",
    FunctionCallFromLLM.__name__: "function_call_from_llm",
    FunctionCallInProgressFrame.__name__: "function_call_in_progress",
    FunctionCallResultFrame.__name__: "function_call_result",
    LLMContextFrame.__name__: "llm_context",
    LLMMessagesFrame.__name__: "llm_messages",
}

SERVICE_TYPE_MAP = {
    STTService.__name__: "stt",
    LLMService.__name__: "llm",
    TTSService.__name__: "tts",
    ImageGenService.__name__: "image_gen",
    VisionService.__name__: "vision",
    WebsocketService.__name__: "websocket",
    AIService.__name__: "ai",
    BaseOutputTransport.__name__: "tts",
}


def get_model_name(service: FrameProcessor) -> str:
    """Get the model name from a service."""
    return str(
        getattr(service, "_full_model_name", None)
        or getattr(service, "model_name", None)
        or getattr(service, "model", "unknown")
    )


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


def detect_frame_type(frame: Frame) -> str:
    """Detect the type of frame using MRO for inheritance support."""
    # Walk through the Method Resolution Order to find first matching frame type
    for base_class in frame.__class__.__mro__:
        frame_type = FRAME_TYPE_MAP.get(base_class.__name__)
        if frame_type:
            return frame_type
    return "unknown"


def detect_service_type(service: FrameProcessor) -> str:
    """Detect the type of service using MRO for inheritance support."""
    # Walk through the Method Resolution Order to find first matching service type
    for base_class in service.__class__.__mro__:
        service_type = SERVICE_TYPE_MAP.get(base_class.__name__)
        if service_type:
            return service_type
    return "unknown"


def detect_service_type_from_class_string(service: str) -> str:
    """
    Detect the type of service from string. MetricsFrames use a string
    to identify processor, so we use this method to determine service type

    Args:
        service: str, ie `GoogleLLMService`

    Returns:
        Service type
    """
    service_type = "unknown"
    if "STTService" in service:
        service_type = "stt"
    elif "LLMService" in service:
        service_type = "llm"
    elif "TTSService" in service:
        service_type = "tts"
    return service_type


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

    _base_attributes: Dict[str, Any] = {
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
    attributes: Dict[str, Any] = {}

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        attributes = {**self._base_attributes, **self.attributes}
        for attribute, operation in attributes.items():
            # Use safe_extract to prevent individual attribute failures from breaking extraction
            value = safe_extract(lambda: operation(frame))
            if value is not None:
                result[attribute] = value
        return result


class TextFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from text frames (TextFrame, LLMTextFrame, TranscriptionFrame, etc.)."""

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        results: Dict[str, Any] = super().extract_from_frame(frame)
        if hasattr(frame, "text") and frame.text:
            text = frame.text

            # Handle different text frame types
            if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
                # Transcription frames are INPUT (user speech)
                results[SpanAttributes.INPUT_VALUE] = text
                results[AudioAttributes.AUDIO_TRANSCRIPT] = text

                results["llm.input_messages.0.message.role"] = "user"
                results["llm.input_messages.0.message.content"] = text
                results["llm.input_messages.0.message.name"] = "stt_text"

                # Add is_final flag for transcriptions
                if isinstance(frame, TranscriptionFrame):
                    results["transcription.is_final"] = True
                    results[SpanAttributes.INPUT_VALUE] = text
                elif isinstance(frame, InterimTranscriptionFrame):
                    results["transcription.is_final"] = False

            elif isinstance(frame, TTSTextFrame):
                # TTSTextFrame represents input TO the TTS service (text to be synthesized)
                # Note: Don't set INPUT_VALUE here - observer will accumulate streaming chunks
                results["text"] = text  # Match Pipecat native tracing attribute name
                results["text.chunk"] = text  # Raw chunk for accumulation

            elif isinstance(frame, LLMTextFrame):
                # LLMTextFrame represents output FROM the LLM service
                # Note: Don't set OUTPUT_VALUE here - observer will accumulate streaming chunks
                results["text.chunk"] = text  # Raw chunk for accumulation

            elif isinstance(frame, TextFrame):
                # Generic text frame (output)
                results[SpanAttributes.OUTPUT_VALUE] = text
                results["llm.output_messages.0.message.role"] = "user"
                results["llm.output_messages.0.message.content"] = text
                results["llm.output_messages.0.message.name"] = "text"

        return results


# Singleton text frame extractor
_text_frame_extractor = TextFrameExtractor()


class LLMContextFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from an LLM context frame."""

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        if hasattr(frame, "context") and frame.context:
            context = frame.context
            # Extract messages from context
            # Note: context.messages is the public API, context._messages is the internal list
            # Try _messages first (more reliable), then fall back to messages
            messages_list = None
            if hasattr(context, "_messages") and context._messages:
                messages_list = context._messages
                results["llm.messages_count"] = len(context._messages)
            elif hasattr(context, "messages") and context.messages:
                messages_list = context.messages
                results["llm.messages_count"] = len(context.messages)

            if messages_list:
                # Convert messages to serializable format
                serializable_messages = []
                for msg in messages_list:
                    if isinstance(msg, dict):
                        serializable_messages.append(msg)
                    elif hasattr(msg, "role") and hasattr(msg, "content"):
                        # LLMMessage object - convert to dict
                        msg_dict = {
                            "role": (str(msg.role) if hasattr(msg.role, "__str__") else msg.role),
                            "content": (
                                str(msg.content)
                                if not isinstance(msg.content, str)
                                else msg.content
                            ),
                        }
                        if hasattr(msg, "name") and msg.name:
                            msg_dict["name"] = msg.name
                        serializable_messages.append(msg_dict)
                    else:
                        # Fallback: try to extract from object attributes
                        try:
                            msg_dict = {
                                "role": getattr(msg, "role", "unknown"),
                                "content": str(msg),
                            }
                            serializable_messages.append(msg_dict)
                        except Exception as e:
                            logger.debug(f"Could not serialize LLMContext message: {e}")
                            pass

                # Store full message history in flattened format that Arize expects
                if serializable_messages:
                    for index, message in enumerate(serializable_messages):
                        if isinstance(message, dict):
                            results[f"llm.input_messages.{index}.message.role"] = message.get(
                                "role"
                            )
                            results[f"llm.input_messages.{index}.message.content"] = message.get(
                                "content"
                            )
                            if message.get("name"):
                                results[f"llm.input_messages.{index}.message.name"] = message.get(
                                    "name"
                                )

                    # For input.value, only capture the LAST user message (current turn's input)
                    last_user_message = None
                    for msg in reversed(serializable_messages):
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            last_user_message = msg
                            break

                    if last_user_message:
                        # Set input.value to just the content of the current turn's user message
                        content = last_user_message.get("content", "")
                        results[SpanAttributes.INPUT_VALUE] = content

                        # Set message attributes with proper role attribution
                        results[MessageAttributes.MESSAGE_ROLE] = last_user_message.get(
                            "role", "user"
                        )
                        results[MessageAttributes.MESSAGE_CONTENT] = content
                        if last_user_message.get("name"):
                            results[MessageAttributes.MESSAGE_NAME] = last_user_message.get("name")
            # Extract tools if present
            if hasattr(context, "_tools") and context._tools:
                try:
                    tools = context._tools
                    if isinstance(tools, list):
                        results["llm.tools_count"] = len(tools)

                        # Extract tool names
                        tool_names = []
                        for tool in tools:
                            if isinstance(tool, dict) and "name" in tool:
                                tool_names.append(tool["name"])
                            elif hasattr(tool, "name"):
                                tool_names.append(tool.name)
                            elif (
                                isinstance(tool, dict)
                                and "function" in tool
                                and "name" in tool["function"]
                            ):
                                tool_names.append(tool["function"]["name"])

                        if tool_names:
                            results["tools.names"] = ",".join(tool_names)

                        # Serialize full tool definitions (with size limit)
                        try:
                            tools_json = safe_json_dumps(tools)
                            if tools_json and len(tools_json) < 10000:  # 10KB limit
                                results["tools.definitions"] = tools_json
                        except (TypeError, ValueError) as e:
                            logger.debug(f"Could not serialize tool definitions: {e}")

                except (TypeError, AttributeError) as e:
                    logger.debug(f"Could not extract tool information: {e}")

        return results


# Singleton LLM context frame extractor
_llm_context_frame_extractor = LLMContextFrameExtractor()


class LLMMessagesFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from an LLM messages frame."""

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        if hasattr(frame, "context") and frame.context and isinstance(frame.context, LLMContext):
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
                            raw_content = msg.get("content")
                            if isinstance(raw_content, str):
                                content = msg.get("content")
                            elif isinstance(raw_content, dict):
                                content = safe_json_dumps(raw_content)
                            else:
                                content = str(raw_content)
                            messages = {
                                "role": msg.get("role"),
                                "content": content,
                                "name": msg.get("name", ""),
                            }
                            messages_list.append(messages)
                        elif isinstance(msg, LLMSpecificMessage):
                            # Fallback: try to serialize the object
                            messages_list.append(msg.message)

                    # Store full message history for reference
                    for index, message in enumerate(messages_list):
                        if isinstance(message, dict):
                            results[f"llm.input_messages.{index}.message.role"] = message.get(
                                "role"
                            )
                            results[f"llm.input_messages.{index}.message.content"] = message.get(
                                "content"
                            )
                            results[f"llm.input_messages.{index}.message.name"] = message.get(
                                "name"
                            )
                        else:
                            results[f"llm.input_messages.{index}.message.role"] = "unknown"
                            results[f"llm.input_messages.{index}.message.content"] = str(message)
                            results[f"llm.input_messages.{index}.message.name"] = "unknown"
                except (TypeError, ValueError, AttributeError) as e:
                    logger.debug(f"Could not serialize LLMContext messages: {e}")

            # Extract tools if present
            if hasattr(context, "_tools") and context._tools:
                try:
                    tools = context._tools

                    # Get tool count
                    if isinstance(tools, list):
                        results["llm.tools_count"] = len(tools)

                        # Extract tool names as comma-separated list
                        tool_names = []
                        for tool in tools:
                            if isinstance(tool, dict) and "name" in tool:
                                tool_names.append(tool["name"])
                            elif hasattr(tool, "name"):
                                tool_names.append(tool.name)
                            elif (
                                isinstance(tool, dict)
                                and "function" in tool
                                and "name" in tool["function"]
                            ):
                                tool_names.append(tool["function"]["name"])

                        if tool_names:
                            results["tools.names"] = ",".join(tool_names)

                        # Serialize full tool definitions (with size limit)
                        try:
                            tools_json = safe_json_dumps(tools)
                            if tools_json and len(tools_json) < 10000:  # 10KB limit
                                results["tools.definitions"] = tools_json
                        except (TypeError, ValueError) as e:
                            logger.debug(f"Could not serialize tool definitions: {e}")

                except (TypeError, AttributeError) as e:
                    logger.debug(f"Could not extract tool information: {e}")

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
        if hasattr(frame, "_messages") and frame._messages:
            messages = frame._messages
            results[SpanAttributes.LLM_INPUT_MESSAGES] = []
            # Extract text content for input.value
            for message in messages:
                if isinstance(message, dict):
                    results[SpanAttributes.LLM_INPUT_MESSAGES].append(
                        {
                            MessageAttributes.MESSAGE_ROLE: message.get("role"),
                            MessageAttributes.MESSAGE_CONTENT: message.get("content"),
                            MessageAttributes.MESSAGE_NAME: message.get("name"),
                        }
                    )
                else:
                    results[SpanAttributes.LLM_INPUT_MESSAGES].append(
                        {
                            MessageAttributes.MESSAGE_ROLE: "unknown",
                            MessageAttributes.MESSAGE_CONTENT: str(message),
                            MessageAttributes.MESSAGE_NAME: "unknown",
                        }
                    )
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


class FunctionCallResultFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from function call result frames."""

    attributes: Dict[str, Any] = {
        SpanAttributes.TOOL_NAME: lambda frame: getattr(frame, "function_name", None),
        ToolCallAttributes.TOOL_CALL_ID: lambda frame: getattr(frame, "tool_call_id", None),
        ToolCallAttributes.TOOL_CALL_FUNCTION_NAME: lambda frame: getattr(
            frame, "function_name", None
        ),
        ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON: lambda frame: (
            safe_json_dumps(getattr(frame, "arguments", {}))
        ),
        "tool.result": lambda frame: (safe_json_dumps(getattr(frame, "result", {}))),
    }


# Singleton function call result frame extractor
_function_call_result_frame_extractor = FunctionCallResultFrameExtractor()


class MetricsDataExtractor:
    """Extract attributes from metrics frames."""

    attributes: Dict[str, Any] = {}
    _base_attributes: Dict[str, Any] = {
        "metrics.processor": lambda metrics_data: getattr(metrics_data, "processor", None),
        "metrics.model": lambda metrics_data: getattr(metrics_data, "model", None),
    }

    def extract_from_metrics_data(self, metrics_data: MetricsData) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        attributes = {**self._base_attributes, **self.attributes}
        for attribute, operation in attributes.items():
            value = safe_extract(lambda: operation(metrics_data))
            if value is not None:
                results[attribute] = value
        return results


class LLMTokenMetricsDataExtractor:
    """Extract attributes from LLM token metrics data."""

    attributes: Dict[str, Any] = {
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: lambda metrics_data: getattr(
            metrics_data, "prompt_tokens", None
        ),
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: lambda metrics_data: getattr(
            metrics_data, "completion_tokens", None
        ),
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: lambda metrics_data: getattr(
            metrics_data, "total_tokens", None
        ),
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: lambda metrics_data: getattr(
            metrics_data, "cache_read_input_tokens", None
        ),
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO: lambda metrics_data: getattr(
            metrics_data, "audio_tokens", None
        ),
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING: lambda metrics_data: getattr(
            metrics_data, "reasoning_tokens", None
        ),
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO: lambda metrics_data: getattr(
            metrics_data, "audio_tokens", None
        ),
    }

    def extract_from_metrics_data(self, metrics_data: LLMTokenUsage) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for attribute, operation in self.attributes.items():
            value = safe_extract(lambda: operation(metrics_data))
            if value is not None:
                results[attribute] = value
        return results


# Singleton LLM token metrics data extractor
_llm_token_metrics_data_extractor = LLMTokenMetricsDataExtractor()


class LLMUsageMetricsDataExtractor(MetricsDataExtractor):
    """Extract attributes from LLM usage metrics data."""

    def extract_from_metrics_data(self, metrics_data: MetricsData) -> Dict[str, Any]:
        results: Dict[str, Any] = super().extract_from_metrics_data(metrics_data)
        if isinstance(metrics_data, LLMUsageMetricsData):
            llm_usage_metrics_data: LLMTokenUsage = metrics_data.value
            results.update(
                _llm_token_metrics_data_extractor.extract_from_metrics_data(llm_usage_metrics_data)
            )
        return results


# Singleton LLM usage metrics data extractor
_llm_usage_metrics_data_extractor = LLMUsageMetricsDataExtractor()


class TTSUsageMetricsDataExtractor(MetricsDataExtractor):
    """Extract attributes from TTS usage metrics data."""

    attributes: Dict[str, Any] = {
        "tts.character_count": lambda metrics_data: getattr(metrics_data, "value", None),
    }


# Singleton TTS usage metrics data extractor
_tts_usage_metrics_data_extractor = TTSUsageMetricsDataExtractor()


class TTFBMetricsDataExtractor(MetricsDataExtractor):
    """Extract attributes from TTFB metrics data."""

    attributes: Dict[str, Any] = {
        "service.ttfb_seconds": lambda metrics_data: getattr(metrics_data, "value", None),
    }


# Singleton TTFB metrics data extractor
_ttfb_metrics_data_extractor = TTFBMetricsDataExtractor()


class ProcessingMetricsDataExtractor(MetricsDataExtractor):
    """Extract attributes from processing metrics data."""

    attributes: Dict[str, Any] = {
        "service.processing_time_seconds": lambda metrics_data: getattr(
            metrics_data, "value", None
        ),
    }


# Singleton processing metrics data extractor
_processing_metrics_data_extractor = ProcessingMetricsDataExtractor()


class MetricsFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from metrics frames."""

    metrics_extractor_map: Dict[Type[MetricsData], MetricsDataExtractor] = {
        LLMUsageMetricsData: _llm_usage_metrics_data_extractor,
        TTSUsageMetricsData: _tts_usage_metrics_data_extractor,
        TTFBMetricsData: _ttfb_metrics_data_extractor,
        ProcessingMetricsData: _processing_metrics_data_extractor,
    }

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        if not hasattr(frame, "data") or not frame.data:
            return results

        metrics: List[MetricsData] = frame.data
        for metrics_data in metrics:
            for base_class in metrics_data.__class__.__mro__:
                extractor = self.metrics_extractor_map.get(base_class)
                if extractor:
                    results.update(extractor.extract_from_metrics_data(metrics_data))
                    break  # Only extract attributes from the first matching extractor
        return results


# Singleton metrics frame extractor
_metrics_frame_extractor = MetricsFrameExtractor()


class GenericFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from a generic frame."""

    frame_extractor_map: Dict[Type[Frame], FrameAttributeExtractor] = {
        TextFrame: _text_frame_extractor,
        LLMContextFrame: _llm_context_frame_extractor,
        LLMMessagesFrame: _llm_messages_frame_extractor,
        LLMMessagesAppendFrame: _llm_messages_append_frame_extractor,
        LLMFullResponseStartFrame: _llm_full_response_start_frame_extractor,
        LLMFullResponseEndFrame: _llm_full_response_end_frame_extractor,
        FunctionCallResultFrame: _function_call_result_frame_extractor,
        MetricsFrame: _metrics_frame_extractor,
    }

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for base_class in frame.__class__.__mro__:
            extractor = self.frame_extractor_map.get(base_class)
            if extractor:
                results.update(extractor.extract_from_frame(frame))
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
    _base_attributes: Dict[str, Any] = {}

    def extract_from_service(self, service: FrameProcessor) -> Dict[str, Any]:
        """Extract attributes from a service."""
        result: Dict[str, Any] = {}
        attributes = {**self._base_attributes, **self.attributes}
        for attribute, operation in attributes.items():
            # Use safe_extract to prevent individual attribute failures from breaking extraction
            value = safe_extract(lambda: operation(service)) if operation else None
            if value is not None:
                result[attribute] = value
        return result


class BaseServiceAttributeExtractor(ServiceAttributeExtractor):
    """Extract base attributes common to all services."""

    _base_attributes: Dict[str, Any] = {
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
        SpanAttributes.LLM_MODEL_NAME: lambda service: get_model_name(service),
        SpanAttributes.LLM_PROVIDER: lambda service: detect_provider_from_service(service),
        # GenAI semantic conventions (dual attributes)
        "gen_ai.system": lambda service: detect_provider_from_service(service),
        "gen_ai.request.model": lambda service: get_model_name(service),
        "gen_ai.operation.name": lambda service: "chat",
        "gen_ai.output.type": lambda service: "text",
        # Streaming flag
        "stream": lambda service: getattr(service, "_stream", True),
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
            OpenInferenceSpanKindValues.LLM.value
        ),
        SpanAttributes.LLM_MODEL_NAME: lambda service: get_model_name(service),
        SpanAttributes.LLM_PROVIDER: lambda service: detect_provider_from_service(service),
        "service.model": lambda service: get_model_name(service),
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
            OpenInferenceSpanKindValues.LLM.value
        ),
        SpanAttributes.LLM_MODEL_NAME: lambda service: get_model_name(service),
        SpanAttributes.LLM_PROVIDER: lambda service: detect_provider_from_service(service),
        "service.model": lambda service: get_model_name(service),
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
        "service.model": lambda service: get_model_name(service),
    }


# Singleton image gen service attribute extractor
_image_gen_service_attribute_extractor = ImageGenServiceAttributeExtractor()


class VisionServiceAttributeExtractor(ServiceAttributeExtractor):
    """Extract attributes from a vision service for span creation."""

    attributes: Dict[str, Any] = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: lambda service: (
            OpenInferenceSpanKindValues.CHAIN.value
        ),
        "service.model": lambda service: get_model_name(service),
    }


# Singleton vision service attribute extractor
_vision_service_attribute_extractor = VisionServiceAttributeExtractor()


class GenericServiceAttributeExtractor(ServiceAttributeExtractor):
    """Extract attributes from a generic service for span creation."""

    service_attribute_extractor_map: Dict[Type[FrameProcessor], ServiceAttributeExtractor] = {
        LLMService: _llm_service_attribute_extractor,
        STTService: _stt_service_attribute_extractor,
        TTSService: _tts_service_attribute_extractor,
        ImageGenService: _image_gen_service_attribute_extractor,
        VisionService: _vision_service_attribute_extractor,
    }

    def extract_from_service(self, service: FrameProcessor) -> Dict[str, Any]:
        """Extract attributes from a generic service."""
        results: Dict[str, Any] = {}
        for base_class in service.__class__.__mro__:
            extractor = self.service_attribute_extractor_map.get(base_class)
            if extractor:
                results.update(extractor.extract_from_service(service))
        return results


# Singleton generic service attribute extractor
_generic_service_attribute_extractor = GenericServiceAttributeExtractor()


def extract_service_attributes(service: FrameProcessor) -> Dict[str, Any]:
    """Extract attributes from a service using the singleton extractor."""
    return _generic_service_attribute_extractor.extract_from_service(service)
