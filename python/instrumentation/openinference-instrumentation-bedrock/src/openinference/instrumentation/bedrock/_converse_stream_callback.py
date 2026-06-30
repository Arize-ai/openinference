from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, TypeVar, cast

from opentelemetry.trace import Span, Status, StatusCode

from openinference.instrumentation.bedrock._converse_attributes import (
    get_attributes_from_response_data,
)

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import (
        ConverseResponseTypeDef,
        ConverseStreamOutputTypeDef,
        ConverseStreamRequestTypeDef,
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
        request: ConverseStreamRequestTypeDef,
    ) -> None:
        self._span = span
        self.role: str = ""
        self.request: ConverseStreamRequestTypeDef = request
        self.message: Dict[str, Any] = {}
        self.stop_reason: str = ""
        self.usage: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.error: Dict[str, Any] = {}

    def _handle_message_start(self, obj: ConverseStreamOutputTypeDef) -> None:
        """Extracts the role from a message start event."""
        if "messageStart" in obj:
            self.role = obj["messageStart"]["role"]

    def _handle_message_stop(self, obj: ConverseStreamOutputTypeDef) -> None:
        """Extracts the stop reason from a message stop event."""
        if "messageStop" in obj:
            self.stop_reason = obj["messageStop"]["stopReason"]

    def _handle_content_block(self, obj: ConverseStreamOutputTypeDef) -> None:
        """Handles content block deltas and updates message content accordingly."""
        content: List[Dict[str, Any]] = self.message.setdefault("content", [])
        if "contentBlockDelta" in obj:
            content_block_delta = obj["contentBlockDelta"]
            index = content_block_delta["contentBlockIndex"]
            # Ensure the content list is long enough
            while len(content) <= index:
                content.append({})
            delta = content_block_delta["delta"]
            if "text" in delta:
                content[index]["text"] = content[index].get("text", "") + delta["text"]
            if "toolUse" in delta and "input" in delta["toolUse"]:
                tool_use = content[index].setdefault("toolUse", {})
                tool_use["input"] = tool_use.get("input", "") + delta["toolUse"]["input"]
        if "contentBlockStart" in obj:
            content.append(dict(obj["contentBlockStart"]["start"]))
        if "contentBlockStop" in obj:
            index = obj["contentBlockStop"]["contentBlockIndex"]
            if (
                0 <= index < len(content)
                and "toolUse" in content[index]
                and isinstance(content[index]["toolUse"].get("input"), str)
            ):
                try:
                    content[index]["toolUse"]["input"] = json.loads(
                        content[index]["toolUse"]["input"]
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse toolUse input as JSON: {e}")

    def _handle_metadata(self, obj: ConverseStreamOutputTypeDef) -> None:
        """Extracts usage, metrics, and error information from metadata."""
        if "metadata" in obj:
            metadata = obj["metadata"]
            if "usage" in metadata:
                self.usage = dict(metadata["usage"])
            if "metrics" in metadata:
                self.metrics = dict(metadata["metrics"])
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
        if isinstance(obj, dict):
            try:
                event = cast("ConverseStreamOutputTypeDef", obj)
                self._handle_content_block(event)
                self._handle_metadata(event)
                self._handle_message_start(event)
                self._handle_message_stop(event)
            except Exception:
                logger.warning("Failed to process converse stream event", exc_info=True)
        elif isinstance(obj, (StopIteration, StopAsyncIteration)):
            try:
                self._span.set_attributes(
                    get_attributes_from_response_data(
                        self.request,
                        cast("ConverseResponseTypeDef", self._construct_final_message()),
                    )
                )
                self._span.set_status(Status(StatusCode.OK))
            except Exception:
                logger.warning("Failed to set response attributes on span", exc_info=True)
            finally:
                self._span.end()
        elif isinstance(obj, BaseException):
            try:
                self._span.record_exception(obj)
                self._span.set_status(Status(StatusCode.ERROR, str(obj)))
            except Exception:
                logger.warning("Failed to record exception on span", exc_info=True)
            finally:
                self._span.end()
        return obj
