from __future__ import annotations

import json
from typing import Any, Iterable

from openinference.instrumentation import Message, ToolCall


def oi_messages(items: Iterable[Any]) -> list[Message]:
    """Convert LiveKit ChatContext items to OpenInference messages.

    A FunctionCall joins the assistant message right before it, or starts one, and a
    FunctionCallOutput becomes a tool message.
    """
    messages: list[Message] = []
    for item in items:
        kind = getattr(item, "type", None)
        if kind == "message":
            message = Message(role=_role(item.role))
            if text := item.text_content:
                message["content"] = text
            messages.append(message)
        elif kind == "function_call":
            call = ToolCall(
                id=item.call_id, function={"name": item.name, "arguments": item.arguments}
            )
            previous = messages[-1] if messages else None
            if previous is None or previous.get("role") != "assistant":
                previous = Message(role="assistant")
                messages.append(previous)
            previous["tool_calls"] = [*previous.get("tool_calls", ()), call]
        elif kind == "function_call_output":
            messages.append(Message(role="tool", tool_call_id=item.call_id, content=item.output))
    return messages


def messages_json(messages: list[Message]) -> str:
    return json.dumps(messages, default=str, ensure_ascii=False)


def _role(role: str) -> str:
    return "system" if role == "developer" else role
