"""Attribute extraction from Pipecat frames."""

import base64
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from openinference.semconv.trace import SpanAttributes
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

logger = logging.getLogger(__name__)

__all__ = [
    "extract_attributes_from_frame",
]


def safe_json_dumps(obj: Any, default: Optional[str] = None) -> Optional[str]:
    """
    Safely serialize an object to JSON, returning None if serialization fails.

    Args:
        obj: The object to serialize
        default: Default value to return on error (defaults to None)

    Returns:
        JSON string or default value on error
    """
    try:
        return json.dumps(obj)
    except Exception as e:
        logger.debug(f"Failed to serialize object to JSON: {e}")
        return default


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
        "text.skip_tts": lambda frame: (
            frame.skip_tts if hasattr(frame, "skip_tts") else None
        ),
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
            len(frame.context._messages)
            if hasattr(frame.context, "_messages")
            else None
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
                results[SpanAttributes.TOOL_PARAMETERS] = safe_extract(
                    lambda: str(frame.arguments)
                )
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
            else str(frame.result) if hasattr(frame, "result") else None
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
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: lambda frame: getattr(
            frame, "prompt_tokens", None
        ),
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: lambda frame: getattr(
            frame, "completion_tokens", None
        ),
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: lambda frame: getattr(
            frame, "total_tokens", None
        ),
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
        "frame.transport_source": lambda frame: getattr(
            frame, "transport_source", None
        ),
        "frame.transport_destination": lambda frame: getattr(
            frame, "transport_destination", None
        ),
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
            results.update(
                _llm_messages_append_frame_extractor.extract_from_frame(frame)
            )
        if isinstance(frame, LLMFullResponseStartFrame):
            results.update(
                _llm_full_response_start_frame_extractor.extract_from_frame(frame)
            )
        if isinstance(frame, LLMFullResponseEndFrame):
            results.update(
                _llm_full_response_end_frame_extractor.extract_from_frame(frame)
            )
        if isinstance(frame, FunctionCallFromLLM):
            results.update(
                _function_call_from_llm_frame_extractor.extract_from_frame(frame)
            )
        if isinstance(frame, FunctionCallResultFrame):
            results.update(
                _function_call_result_frame_extractor.extract_from_frame(frame)
            )
        if isinstance(frame, FunctionCallInProgressFrame):
            results.update(
                _function_call_in_progress_frame_extractor.extract_from_frame(frame)
            )
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
