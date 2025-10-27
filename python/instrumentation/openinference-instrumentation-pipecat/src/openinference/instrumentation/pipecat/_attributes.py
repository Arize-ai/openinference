"""Attribute extraction from Pipecat frames."""

from typing import Any, Dict, Optional

from openinference.semconv.trace import SpanAttributes


class _FrameAttributeExtractor:
    """Extract attributes from Pipecat frames."""

    def __init__(self, max_length: int = 1000):
        """
        Initialize extractor.

        Args:
            max_length: Maximum length for text values
        """
        self._max_length = max_length

    def extract_from_frame(self, frame) -> Dict[str, Any]:
        """
        Extract attributes from a frame.

        Args:
            frame: A Pipecat frame

        Returns:
            Dictionary of attributes
        """
        attributes = {}

        try:
            from pipecat.frames.frames import (
                LLMMessagesFrame,
                TextFrame,
                TranscriptionFrame,
            )

            # TextFrame -> INPUT_VALUE
            if isinstance(frame, TextFrame):
                if hasattr(frame, "text") and frame.text:
                    attributes[SpanAttributes.INPUT_VALUE] = self._truncate(frame.text)

            # TranscriptionFrame -> OUTPUT_VALUE (STT output)
            elif isinstance(frame, TranscriptionFrame):
                if hasattr(frame, "text") and frame.text:
                    attributes[SpanAttributes.OUTPUT_VALUE] = self._truncate(frame.text)

            # LLMMessagesFrame -> INPUT_VALUE
            elif isinstance(frame, LLMMessagesFrame):
                if hasattr(frame, "messages") and frame.messages:
                    # Extract last user message
                    for msg in reversed(frame.messages):
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            content = msg.get("content", "")
                            attributes[SpanAttributes.INPUT_VALUE] = self._truncate(
                                str(content)
                            )
                            break

        except (ImportError, AttributeError):
            pass

        return attributes

    def _truncate(self, text: str) -> str:
        """Truncate text to max_length."""
        if text is None:
            return ""
        text = str(text)
        if len(text) <= self._max_length:
            return text
        return text[: self._max_length]
