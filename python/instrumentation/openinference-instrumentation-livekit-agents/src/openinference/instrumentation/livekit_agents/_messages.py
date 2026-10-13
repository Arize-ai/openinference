from __future__ import annotations

import json
from typing import Any

from openinference.instrumentation import Message, ToolCall


def oi_messages(
    messages_json: str | None,
    system_instructions_json: str | None = None,
) -> list[Message]:
    """Convert LiveKit's gen_ai `{role, parts}` messages to OpenInference messages."""
    messages: list[Message] = []
    system = _text(_load(system_instructions_json))
    if system:
        messages.append(Message(role="system", content=system))
    for item in _load(messages_json):
        if isinstance(item, dict):
            messages.append(_message(item))
    return messages


def _message(item: dict[str, Any]) -> Message:
    message = Message()
    if isinstance(role := item.get("role"), str):
        message["role"] = role
    parts = [part for part in item.get("parts") or () if isinstance(part, dict)]
    if text := _text(parts):
        message["content"] = text
    tool_calls: list[ToolCall] = []
    for part in parts:
        if part.get("type") == "tool_call":
            tool_call = ToolCall(function={})
            if isinstance(call_id := part.get("id"), str):
                tool_call["id"] = call_id
            if isinstance(name := part.get("name"), str):
                tool_call["function"]["name"] = name
            if (arguments := part.get("arguments")) is not None:
                tool_call["function"]["arguments"] = (
                    arguments if isinstance(arguments, str) else json.dumps(arguments)
                )
            tool_calls.append(tool_call)
        elif part.get("type") == "tool_call_response":
            if isinstance(call_id := part.get("id"), str):
                message["tool_call_id"] = call_id
            if "content" not in message and (response := part.get("response")) is not None:
                message["content"] = response if isinstance(response, str) else json.dumps(response)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _text(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    return "".join(
        part["content"]
        for part in parts
        if isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("content"), str)
    )


def _load(value: str | None) -> list[Any]:
    if value is None:
        return []
    try:
        loaded = json.loads(value)
    except ValueError:
        return []
    return loaded if isinstance(loaded, list) else []
