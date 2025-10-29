"""Attribute extraction from Pipecat frames."""

import logging
import json
from typing import Any, Dict, List, Optional

from openinference.semconv.trace import SpanAttributes
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    LLMMessagesFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TTSAudioRawFrame,
    AudioRawFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    UserAudioRawFrame,
    FunctionCallFromLLM,
    FunctionCallResultFrame,
    FunctionCallInProgressFrame,
    ErrorFrame,
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
        attributes = {}

        # ALWAYS capture frame type
        attributes["frame.type"] = frame.__class__.__name__

        # Pattern 1: Text content (TextFrame, TranscriptionFrame, etc.)
        try:
            if hasattr(frame, "text") and frame.text:
                # For transcription, this is output from STT
                if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
                    attributes[SpanAttributes.OUTPUT_VALUE] = frame.text
                # For text frames going to TTS/LLM, this is input
                else:
                    attributes[SpanAttributes.INPUT_VALUE] = frame.text
        except (TypeError, ValueError):
            logger.error(f"Error extracting text from frame: {frame}")
            pass

        # Pattern 2: Audio metadata (AudioRawFrame variants)
        try:
            if hasattr(frame, "sample_rate") and frame.sample_rate:
                attributes["audio.sample_rate"] = frame.sample_rate
            if hasattr(frame, "num_channels") and frame.num_channels:
                attributes["audio.num_channels"] = frame.num_channels
            if hasattr(frame, "audio") and frame.audio:
                # Don't store actual audio data, just indicate presence and size
                attributes["audio.size_bytes"] = len(frame.audio)
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

        return attributes

    def _extract_llm_attributes(self, frame: Frame) -> Dict[str, Any]:
        """
        Extract LLM-specific attributes from LLM frames.

        Handles: LLMMessagesFrame, LLMMessagesAppendFrame, LLMFullResponseStartFrame, etc.
        """
        attributes = {}

        # LLMMessagesFrame contains the full message history
        try:
            if isinstance(frame, LLMMessagesFrame):
                if hasattr(frame, "messages") and frame.messages:
                    attributes["llm.messages_count"] = len(frame.messages)

                    # Extract text content for input.value
                    user_messages = [msg.get("content", "") for msg in frame.messages]
                    if user_messages:
                        attributes[SpanAttributes.INPUT_VALUE] = json.dumps(
                            user_messages
                        )
            # LLMMessagesAppendFrame adds messages to context
            elif isinstance(frame, LLMMessagesAppendFrame):
                if hasattr(frame, "messages") and frame.messages:
                    attributes["llm.messages_appended"] = len(frame.messages)

            # LLM response boundaries
            elif isinstance(frame, LLMFullResponseStartFrame):
                attributes["llm.response_phase"] = "start"
            elif isinstance(frame, LLMFullResponseEndFrame):
                attributes["llm.response_phase"] = "end"
        except (TypeError, ValueError):
            logger.error(f"Error extracting LLM attributes from frame: {frame}")
            pass
        finally:
            return attributes

    def _extract_tool_attributes(self, frame: Frame) -> Dict[str, Any]:
        """Extract function calling / tool use attributes."""
        attributes = {}

        # Function call from LLM
        try:
            if isinstance(frame, FunctionCallFromLLM):
                if hasattr(frame, "function_name") and frame.function_name:
                    attributes[SpanAttributes.TOOL_NAME] = frame.function_name
                if hasattr(frame, "arguments") and frame.arguments:

                    # Arguments are typically a dict
                    if isinstance(frame.arguments, dict):
                        attributes[SpanAttributes.TOOL_PARAMETERS] = json.dumps(
                            frame.arguments
                        )
                    else:
                        attributes[SpanAttributes.TOOL_PARAMETERS] = str(
                            frame.arguments
                        )
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
