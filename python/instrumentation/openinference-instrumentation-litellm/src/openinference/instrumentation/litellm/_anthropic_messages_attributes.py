"""Attribute helpers for LiteLLM Anthropic Messages API (create / acreate)."""

from __future__ import annotations

import codecs
import json
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolCallAttributes,
)

ANTHROPIC_KEYS_TO_REDACT = ["api_key", "messages", "system"]


def _get_block_type(block: Any) -> Optional[str]:
    if isinstance(block, Mapping):
        type_ = block.get("type")
        return str(type_) if type_ is not None else None
    return getattr(block, "type", None)


def _get_block_field(block: Any, field: str) -> Any:
    if isinstance(block, Mapping):
        return block.get(field)
    return getattr(block, field, None)


def _get_attributes_from_anthropic_content_blocks(
    content: Any,
    message_prefix: str,
) -> Iterator[Tuple[str, AttributeValue]]:
    """Yield OI attributes for Anthropic content blocks on an input/output message."""
    if isinstance(content, str):
        yield f"{message_prefix}.{MessageAttributes.MESSAGE_CONTENT}", content
        return

    if not isinstance(content, Iterable) or isinstance(content, (str, bytes)):
        return

    tool_index = 0
    for j, block in enumerate(content):
        block_type = _get_block_type(block)
        if block_type is None:
            continue
        prefix = f"{message_prefix}.{MessageAttributes.MESSAGE_CONTENTS}.{j}"

        if block_type == "text":
            yield f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text"
            if text := _get_block_field(block, "text"):
                yield f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", text
        elif block_type == "tool_use":
            tool_id = _get_block_field(block, "id")
            tool_name = _get_block_field(block, "name")
            tool_input = _get_block_field(block, "input")
            if tool_id is not None:
                yield (
                    f"{message_prefix}.{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_index}.{ToolCallAttributes.TOOL_CALL_ID}",
                    tool_id,
                )
                yield f"{prefix}.{ToolCallAttributes.TOOL_CALL_ID}", tool_id
            if tool_name is not None:
                yield (
                    f"{message_prefix}.{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_index}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                    tool_name,
                )
                yield f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}", tool_name
            if tool_input is not None:
                args_json = (
                    tool_input if isinstance(tool_input, str) else safe_json_dumps(tool_input)
                )
                yield (
                    f"{message_prefix}.{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_index}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                    args_json,
                )
                yield f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}", args_json
            yield f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "tool_use"
            tool_index += 1
        elif block_type == "tool_result":
            if tool_use_id := _get_block_field(block, "tool_use_id"):
                yield f"{message_prefix}.{MessageAttributes.MESSAGE_TOOL_CALL_ID}", tool_use_id
            if (tool_result_content := _get_block_field(block, "content")) is not None:
                yield (
                    f"{message_prefix}.{MessageAttributes.MESSAGE_CONTENT}",
                    tool_result_content
                    if isinstance(tool_result_content, str)
                    else safe_json_dumps(tool_result_content),
                )
        elif block_type == "image":
            source = _get_block_field(block, "source")
            if isinstance(source, Mapping):
                media_type = source.get("media_type", "")
                source_type = source.get("type", "")
                data = source.get("data", "")
                image_data = f"data:{media_type};{source_type},{data}"
                yield f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "image"
                yield (
                    f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}",
                    image_data,
                )
        elif block_type == "thinking":
            yield f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "reasoning"
            if thinking := _get_block_field(block, "thinking"):
                yield f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", thinking
            if signature := _get_block_field(block, "signature"):
                yield (
                    f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE}",
                    signature,
                )
        elif block_type == "redacted_thinking":
            yield f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "reasoning"
            if data := _get_block_field(block, "data"):
                yield f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_DATA}", data


def _get_attributes_from_anthropic_input_messages(
    messages: Iterable[Mapping[str, Any]],
    system: Optional[Union[str, Iterable[Mapping[str, Any]]]] = None,
) -> Iterator[Tuple[str, AttributeValue]]:
    """
    Extract input message attributes.

    Anthropic Messages API takes ``system`` as a top-level param; prepend a
    synthetic system message so OI consumers see it in LLM_INPUT_MESSAGES.
    """
    start = 1 if system else 0
    for i, message in enumerate(messages, start=start):
        prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.{i}"
        if role := message.get("role"):
            yield f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", role
        if (content := message.get("content")) is not None:
            yield from _get_attributes_from_anthropic_content_blocks(content, prefix)

    if system:
        prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0"
        yield f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", "system"
        if isinstance(system, str):
            yield f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}", system
        else:
            yield from _get_attributes_from_anthropic_content_blocks(system, prefix)


def _get_attributes_from_anthropic_output_message(
    response: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    if role := response.get("role"):
        yield f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", role
    if (content := response.get("content")) is not None:
        yield from _get_attributes_from_anthropic_content_blocks(content, prefix)


def _get_output_text_from_anthropic_response(response: Mapping[str, Any]) -> Optional[str]:
    content = response.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, Iterable) or isinstance(content, (str, bytes)):
        return None
    texts: List[str] = []
    for block in content:
        if _get_block_type(block) == "text":
            if text := _get_block_field(block, "text"):
                texts.append(str(text))
    if not texts:
        return None
    return "".join(texts) if len(texts) == 1 else "\n".join(texts)


def _normalize_stream_event(chunk: Any) -> Optional[Dict[str, Any]]:
    """Normalize a streaming chunk into an Anthropic event dict when possible."""
    if isinstance(chunk, Mapping):
        return dict(chunk)
    if hasattr(chunk, "model_dump"):
        try:
            dumped = chunk.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if isinstance(chunk, (bytes, bytearray)):
        try:
            text = chunk.decode("utf-8")
        except Exception:
            return None
        return _parse_sse_payload(text)
    if isinstance(chunk, str):
        return _parse_sse_payload(chunk)
    event_type = getattr(chunk, "type", None)
    if event_type is not None:
        event: Dict[str, Any] = {"type": event_type}
        for field in ("delta", "content_block", "message", "index", "usage"):
            if (value := getattr(chunk, field, None)) is not None:
                if hasattr(value, "model_dump"):
                    try:
                        value = value.model_dump()
                    except Exception:
                        pass
                event[field] = value
        return event
    return None


def _parse_sse_payload(text: str) -> Optional[Dict[str, Any]]:
    data_lines: List[str] = []
    for line in text.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if not data_lines:
        stripped = text.strip()
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None
    payload = "\n".join(data_lines)
    if payload == "[DONE]":
        return None
    try:
        parsed = json.loads(payload)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


class AnthropicMessagesStreamAccumulator:
    """Accumulate Anthropic Messages streaming events into a final response shape."""

    def __init__(self) -> None:
        self.role: str = "assistant"
        self.model: Optional[str] = None
        self.content_blocks: Dict[int, Dict[str, Any]] = {}
        self.usage: Dict[str, Any] = {}
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._sse_buffer = ""

    def process_chunk(self, chunk: Any) -> None:
        if isinstance(chunk, (bytes, bytearray)):
            self._sse_buffer += self._decoder.decode(bytes(chunk))
            self._process_complete_sse_events()
            return
        if isinstance(chunk, str):
            self._sse_buffer += chunk
            self._process_complete_sse_events()
            return
        self._process_event(_normalize_stream_event(chunk))

    def finish(self) -> None:
        self._sse_buffer += self._decoder.decode(b"", final=True)
        self._process_complete_sse_events(final=True)

    def _process_complete_sse_events(self, *, final: bool = False) -> None:
        normalized = self._sse_buffer.replace("\r\n", "\n")
        events = normalized.split("\n\n")
        self._sse_buffer = "" if final else events.pop()
        for payload in events:
            self._process_event(_parse_sse_payload(payload))
        if final and self._sse_buffer:
            self._process_event(_parse_sse_payload(self._sse_buffer))
            self._sse_buffer = ""

    def _process_event(self, event: Optional[Dict[str, Any]]) -> None:
        if event is None:
            return
        event_type = event.get("type")
        if event_type == "message_start":
            message = event.get("message") or {}
            if isinstance(message, Mapping):
                if role := message.get("role"):
                    self.role = str(role)
                if model := message.get("model"):
                    self.model = str(model)
                if usage := message.get("usage"):
                    if isinstance(usage, Mapping):
                        self.usage.update(dict(usage))
        elif event_type == "content_block_start":
            index = event.get("index", 0)
            block = event.get("content_block") or {}
            if isinstance(block, Mapping):
                self.content_blocks[index] = dict(block)
        elif event_type == "content_block_delta":
            index = event.get("index", 0)
            delta = event.get("delta") or {}
            if not isinstance(delta, Mapping):
                return
            entry = self.content_blocks.setdefault(index, {"type": delta.get("type", "text")})
            delta_type = delta.get("type")
            if delta_type == "text_delta" or "text" in delta:
                entry["type"] = "text"
                entry["text"] = entry.get("text", "") + str(delta.get("text", ""))
            elif delta_type == "input_json_delta":
                entry["type"] = entry.get("type") or "tool_use"
                entry["input_json"] = entry.get("input_json", "") + str(
                    delta.get("partial_json", "")
                )
            elif delta_type == "thinking_delta":
                entry["type"] = "thinking"
                entry["thinking"] = entry.get("thinking", "") + str(delta.get("thinking", ""))
            elif delta_type == "signature_delta":
                entry["type"] = "thinking"
                entry["signature"] = entry.get("signature", "") + str(delta.get("signature", ""))
        elif event_type == "message_delta":
            if usage := event.get("usage"):
                if isinstance(usage, Mapping):
                    self.usage.update(dict(usage))
            delta = event.get("delta") or {}
            if isinstance(delta, Mapping) and (usage := delta.get("usage")):
                if isinstance(usage, Mapping):
                    self.usage.update(dict(usage))

    def to_response(self) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        for index in sorted(self.content_blocks.keys()):
            block = dict(self.content_blocks[index])
            if "input_json" in block:
                raw = block.pop("input_json")
                try:
                    block["input"] = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    block["input"] = {"_partial_json": raw}
            content.append(block)
        response: Dict[str, Any] = {
            "role": self.role,
            "type": "message",
            "content": content,
        }
        if self.model:
            response["model"] = self.model
        if self.usage:
            response["usage"] = self.usage
        return response
