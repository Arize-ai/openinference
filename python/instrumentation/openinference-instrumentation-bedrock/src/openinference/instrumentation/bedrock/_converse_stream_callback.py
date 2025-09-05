from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, TypeVar

from opentelemetry.trace import Span, Status, StatusCode

from openinference.instrumentation.bedrock._converse_attributes import (
    get_attributes_from_response_data,
)

_AnyT = TypeVar("_AnyT")

logger = logging.getLogger(__name__)


class _ConverseStreamCallback:
    """
    Callback for handling streaming Bedrock responses, accumulating message content, usage,
    metrics, and errors.
    """

    def __init__(
        self,
        span: Span,
        request: Dict[str, Any],
    ) -> None:
        self._span = span
        self.role: str = ""
        self.request: Dict[str, Any] = request
        self.message: Dict[str, Any] = {}
        self.stop_reason: str = ""
        self.usage: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.error: Dict[str, Any] = {}

    def _handle_message_start(self, obj: Dict[str, Any]) -> None:
        """Extracts the role from a message start event."""
        self.role = obj.get("messageStart", {}).get("role", self.role)

    def _handle_message_stop(self, obj: Dict[str, Any]) -> None:
        """Extracts the stop reason from a message stop event."""
        self.stop_reason = obj.get("messageStop", {}).get("stopReason", self.stop_reason)

    def _handle_content_block(self, obj: Dict[str, Any]) -> None:
        """Handles content block deltas and updates message content accordingly."""
        content: List[Dict[str, Any]] = self.message.setdefault("content", [])
        content_block_delta = obj.get("contentBlockDelta", {})
        if content_block_delta:
            index = content_block_delta.get("contentBlockIndex", len(content))
            # Ensure the content list is long enough
            while len(content) <= index:
                content.append({})
            # Handle text delta
            item_text = content_block_delta.get("delta", {}).get("text")
            if item_text:
                content[index]["text"] = content[index].get("text", "") + item_text
            # Handle toolUse input delta
            input_text = content_block_delta.get("delta", {}).get("toolUse", {}).get("input")
            if input_text:
                tool_use = content[index].setdefault("toolUse", {})
                tool_use["input"] = tool_use.get("input", "") + input_text
        # Handle content block start
        content_block_start = obj.get("contentBlockStart", {})
        if content_block_start:
            content.append(content_block_start.get("start", {}))
        # Handle content block stop
        content_block_stop = obj.get("contentBlockStop", {})
        if content_block_stop:
            index = content_block_stop.get("contentBlockIndex")
            if (
                index is not None
                and 0 <= index < len(content)
                and "toolUse" in content[index]
                and isinstance(content[index]["toolUse"].get("input"), str)
            ):
                input_text = content[index]["toolUse"]["input"]
                try:
                    content[index]["toolUse"]["input"] = json.loads(input_text)
                except Exception as e:
                    logger.warning(f"Failed to parse toolUse input as JSON: {e}")

    def _handle_metadata(self, obj: Dict[str, Any]) -> None:
        """Extracts usage, metrics, and error information from metadata."""
        usage = obj.get("metadata", {}).get("usage")
        if usage:
            self.usage = usage
        metrics = obj.get("metadata", {}).get("metrics")
        if metrics:
            self.metrics = metrics
        for key, value in obj.items():
            if "exception" in key:
                self.error[key] = value

    def _construct_final_message(self) -> Dict[str, Any]:
        """Constructs the final output message with all accumulated data."""
        message = {"role": self.role}
        message.update(self.message)
        final_output: Dict[str, Any] = {"output": {"message": message}}
        if self.stop_reason:
            final_output["stopReason"] = self.stop_reason
        if self.usage:
            final_output["usage"] = self.usage
        if self.metrics:
            final_output["metrics"] = self.metrics
        if self.error:
            final_output["output"].update(self.error)
        return final_output

    def __call__(self, obj: _AnyT) -> _AnyT:
        """Processes a streaming event object."""
        try:
            if isinstance(obj, dict):
                self._handle_content_block(obj)
                self._handle_metadata(obj)
                self._handle_message_start(obj)
                self._handle_message_stop(obj)
            elif isinstance(obj, (StopIteration, StopAsyncIteration)):
                self._span.set_attributes(
                    get_attributes_from_response_data(self.request, self._construct_final_message())
                )
                self._span.set_status(Status(StatusCode.OK))
                self._span.end()
        except Exception as e:
            logger.error(f"Error in _ConverseStreamCallback: {e}", exc_info=True)
            self._span.set_status(Status(StatusCode.ERROR, str(e)))
            self._span.end()
        return obj
