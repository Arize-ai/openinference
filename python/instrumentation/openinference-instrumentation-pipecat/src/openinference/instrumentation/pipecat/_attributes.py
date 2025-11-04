"""Attribute extraction from Pipecat frames."""

import base64
import json
import logging
from typing import Any, Dict, List

from openinference.semconv.trace import SpanAttributes
from pipecat.frames.frames import (
    AudioRawFrame,
    ErrorFrame,
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
    LLMMessagesUpdateFrame,
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


class _FrameAttributeExtractor:
    """Extract attributes from Pipecat frames using pattern-based detection."""

    def __init__(self, max_length: int = 1000):
        """
        Initialize extractor.

        Args:
            max_length: Maximum length for text values
        """
        self._max_length = max_length

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        """
        Extract attributes from a frame using pattern-based detection.

        This method handles 100+ Pipecat frame types without creating
        unique handlers for each one. It uses duck-typing to detect
        common properties across frame types.

        Args:
            frame: A Pipecat frame

        Returns:
            Dictionary of attributes following OpenInference conventions
        """
        attributes: Dict[str, Any] = {}

        # ALWAYS capture frame type
        attributes["frame.type"] = frame.__class__.__name__

        # Pattern 1: Text content (TextFrame, TranscriptionFrame, etc.)
        try:
            if isinstance(frame, TextFrame):
                # For transcription, this is output from STT
                attributes["text.skip_tts"] = frame.skip_tts
                if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
                    attributes[SpanAttributes.OUTPUT_VALUE] = frame.text
                else:
                    attributes[SpanAttributes.INPUT_VALUE] = frame.text
        except (TypeError, ValueError):
            logger.error(f"Error extracting text from frame: {frame}")
            pass

        # Pattern 2: Audio metadata (AudioRawFrame variants)
        try:
            if isinstance(frame, AudioRawFrame):
                attributes["audio"] = base64.b64encode(frame.audio).decode("utf-8")
                attributes["audio.sample_rate"] = frame.sample_rate
                attributes["audio.num_channels"] = frame.num_channels
                attributes["audio.size_bytes"] = len(frame.audio)
                attributes["audio.frame_count"] = frame.num_frames
        except (TypeError, ValueError):
            logger.error(f"Error extracting audio metadata from frame: {frame}")
            pass
        # Pattern 3: User metadata (for user attribution)
        try:
            if hasattr(frame, "user_id") and frame.user_id:
                attributes[SpanAttributes.USER_ID] = frame.user_id
        except (TypeError, ValueError):
            logger.error(f"Error extracting user metadata from frame: {frame}")
            pass
        # Pattern 4: Timestamps (for timing analysis)
        try:
            if hasattr(frame, "timestamp") and frame.timestamp is not None:
                attributes["frame.timestamp"] = frame.timestamp
            if hasattr(frame, "pts") and frame.pts is not None:
                attributes["frame.pts"] = frame.pts
        except (TypeError, ValueError):
            logger.error(f"Error extracting metadata from frame: {frame}")
            pass

        # Pattern 5: Error information
        try:
            if isinstance(frame, ErrorFrame):
                if hasattr(frame, "error") and frame.error:
                    attributes["frame.error.message"] = str(frame.error)
        except (TypeError, ValueError):
            logger.error(f"Error extracting error information from frame: {frame}")
            pass

        # Pattern 6: LLM Messages (special handling for LLM frames)
        attributes.update(self._extract_llm_attributes(frame))

        # Pattern 7: Function calling / Tool use
        attributes.update(self._extract_tool_attributes(frame))

        # Pattern 8: Frame metadata (if present)
        if hasattr(frame, "metadata") and frame.metadata:
            # Store as JSON string if it's a dict
            if isinstance(frame.metadata, dict):
                try:
                    attributes["frame.metadata"] = json.dumps(frame.metadata)
                except (TypeError, ValueError):
                    pass

        # Pattern 9: Metrics data (usage, TTFB, processing time)
        attributes.update(self._extract_metrics_attributes(frame))

        return attributes

    def _extract_llm_attributes(self, frame: Frame) -> Dict[str, Any]:
        """
        Extract LLM-specific attributes from LLM frames.

        Handles: LLMContextFrame, LLMMessagesFrame, LLMMessagesAppendFrame,
        LLMFullResponseStartFrame, etc.
        """
        attributes: Dict[str, Any] = {}

        try:
            # LLMContextFrame contains the universal LLM context
            if isinstance(frame, LLMContextFrame):
                if hasattr(frame, "context") and frame.context:
                    context = frame.context
                    # Extract messages from context (context._messages is a list)
                    if hasattr(context, "_messages") and context._messages:
                        attributes["llm.messages_count"] = len(context._messages)

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
                                        content = json.dumps(raw_content)
                                    else:
                                        content = str(raw_content)
                                    messages = {
                                        "role": msg.role,  # type: ignore
                                        "content": content,
                                        "name": msg.name if hasattr(msg, "name") else "",  # type: ignore
                                    }
                                    messages_list.append(messages)
                                elif isinstance(msg, LLMSpecificMessage):
                                    # Fallback: try to serialize the object
                                    messages_list.append(msg.message)
                            messages_json = json.dumps(messages_list)
                            attributes[SpanAttributes.LLM_INPUT_MESSAGES] = messages_json
                            attributes[SpanAttributes.INPUT_VALUE] = messages_json
                        except (TypeError, ValueError, AttributeError) as e:
                            logger.debug(f"Could not serialize LLMContext messages: {e}")

                    # Extract tools if present
                    if hasattr(context, "_tools") and context._tools:
                        try:
                            # Try to get tool count
                            if isinstance(context._tools, list):
                                attributes["llm.tools_count"] = len(context._tools)
                        except (TypeError, AttributeError):
                            pass

            # LLMMessagesFrame and LLMMessagesUpdateFrame contain the full message history
            elif isinstance(frame, (LLMMessagesFrame, LLMMessagesUpdateFrame)):
                if hasattr(frame, "messages") and frame.messages:
                    attributes["llm.messages_count"] = len(frame.messages)

                    # Extract text content for input.value
                    user_messages = json.dumps(frame.messages)
                    attributes[SpanAttributes.LLM_INPUT_MESSAGES] = user_messages
                    attributes[SpanAttributes.INPUT_VALUE] = user_messages
            # LLMMessagesAppendFrame adds messages to context
            elif isinstance(frame, LLMMessagesAppendFrame):
                if hasattr(frame, "messages") and frame.messages:
                    attributes["llm.messages_appended"] = len(frame.messages)

            # LLM response boundaries
            elif isinstance(frame, LLMFullResponseStartFrame):
                attributes["llm.response_phase"] = "start"
                if hasattr(frame, "messages") and frame.messages:
                    attributes["llm.messages_count"] = len(frame.messages)
                    user_messages = json.dumps(frame.messages)
                    attributes[SpanAttributes.LLM_OUTPUT_MESSAGES] = user_messages
            elif isinstance(frame, LLMFullResponseEndFrame):
                attributes["llm.response_phase"] = "end"
                if hasattr(frame, "messages") and frame.messages:
                    attributes["llm.messages_count"] = len(frame.messages)
                    user_messages = json.dumps(frame.messages)
                    attributes[SpanAttributes.LLM_OUTPUT_MESSAGES] = user_messages
        except (TypeError, ValueError):
            logger.error(f"Error extracting LLM attributes from frame: {frame}")
            pass
        finally:
            return attributes

    def _extract_tool_attributes(self, frame: Frame) -> Dict[str, Any]:
        """Extract function calling / tool use attributes."""
        attributes: Dict[str, Any] = {}

        # Function call from LLM
        try:
            if isinstance(frame, FunctionCallFromLLM):
                if hasattr(frame, "function_name") and frame.function_name:
                    attributes[SpanAttributes.TOOL_NAME] = frame.function_name
                if hasattr(frame, "arguments") and frame.arguments:
                    # Arguments are typically a dict
                    if isinstance(frame.arguments, dict):
                        attributes[SpanAttributes.TOOL_PARAMETERS] = json.dumps(frame.arguments)
                    else:
                        attributes[SpanAttributes.TOOL_PARAMETERS] = str(frame.arguments)
                if hasattr(frame, "tool_call_id") and frame.tool_call_id:
                    attributes["tool.call_id"] = frame.tool_call_id

            # Function call result
            elif isinstance(frame, FunctionCallResultFrame):
                if hasattr(frame, "function_name") and frame.function_name:
                    attributes[SpanAttributes.TOOL_NAME] = frame.function_name
                if hasattr(frame, "result") and frame.result:
                    # Result could be any type
                    if isinstance(frame.result, (dict, list)):
                        attributes["tool.result"] = json.dumps(frame.result)
                    else:
                        attributes["tool.result"] = str(frame.result)
                if hasattr(frame, "tool_call_id") and frame.tool_call_id:
                    attributes["tool.call_id"] = frame.tool_call_id

            # In-progress function call
            elif isinstance(frame, FunctionCallInProgressFrame):
                if hasattr(frame, "function_name") and frame.function_name:
                    attributes[SpanAttributes.TOOL_NAME] = frame.function_name
                    attributes["tool.status"] = "in_progress"
        except (TypeError, ValueError):
            logger.error(f"Error extracting tool attributes from frame: {frame}")
            pass
        finally:
            return attributes

    def _extract_metrics_attributes(self, frame: Frame) -> Dict[str, Any]:
        """
        Extract metrics attributes from MetricsFrame.

        Handles: LLMUsageMetricsData, TTSUsageMetricsData, TTFBMetricsData, ProcessingMetricsData
        """
        attributes: Dict[str, Any] = {}

        try:
            if isinstance(frame, MetricsFrame):
                # MetricsFrame contains a list of MetricsData objects
                if hasattr(frame, "data") and frame.data:
                    for metrics_data in frame.data:
                        # LLM token usage metrics
                        if isinstance(metrics_data, LLMUsageMetricsData):
                            if hasattr(metrics_data, "value") and metrics_data.value:
                                token_usage = metrics_data.value
                                if hasattr(token_usage, "prompt_tokens"):
                                    attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = (
                                        token_usage.prompt_tokens
                                    )
                                if hasattr(token_usage, "completion_tokens"):
                                    attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = (
                                        token_usage.completion_tokens
                                    )
                                if hasattr(token_usage, "total_tokens"):
                                    attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] = (
                                        token_usage.total_tokens
                                    )

                                # Optional token fields
                                if (
                                    hasattr(token_usage, "cache_read_input_tokens")
                                    and token_usage.cache_read_input_tokens
                                ):
                                    attributes["llm.token_count.cache_read"] = (
                                        token_usage.cache_read_input_tokens
                                    )
                                if (
                                    hasattr(token_usage, "cache_creation_input_tokens")
                                    and token_usage.cache_creation_input_tokens
                                ):
                                    attributes["llm.token_count.cache_creation"] = (
                                        token_usage.cache_creation_input_tokens
                                    )
                                if (
                                    hasattr(token_usage, "reasoning_tokens")
                                    and token_usage.reasoning_tokens
                                ):
                                    attributes["llm.token_count.reasoning"] = (
                                        token_usage.reasoning_tokens
                                    )

                        # TTS character usage metrics
                        elif isinstance(metrics_data, TTSUsageMetricsData):
                            if hasattr(metrics_data, "value"):
                                attributes["tts.character_count"] = metrics_data.value

                        # Time to first byte metrics
                        elif isinstance(metrics_data, TTFBMetricsData):
                            if hasattr(metrics_data, "value"):
                                attributes["service.ttfb_seconds"] = metrics_data.value

                        # Processing time metrics
                        elif isinstance(metrics_data, ProcessingMetricsData):
                            if hasattr(metrics_data, "value"):
                                attributes["service.processing_time_seconds"] = metrics_data.value

        except (TypeError, ValueError, AttributeError) as e:
            logger.debug(f"Error extracting metrics from frame: {e}")

        return attributes
