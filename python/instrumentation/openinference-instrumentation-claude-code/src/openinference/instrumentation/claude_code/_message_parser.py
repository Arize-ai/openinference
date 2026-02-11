"""Message parsing utilities for extracting span data."""

from typing import Any, Dict, List

from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock


def extract_text_content(message: AssistantMessage) -> str:
    """Extract text content from message content blocks."""
    if not hasattr(message, "content"):
        return ""

    text_parts = []
    for block in message.content:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)

    return "\n".join(text_parts)


def extract_tool_uses(message: AssistantMessage) -> List[Dict[str, Any]]:
    """Extract tool use blocks from message."""
    if not hasattr(message, "content"):
        return []

    tool_uses = []
    for block in message.content:
        if isinstance(block, ToolUseBlock):
            tool_uses.append(
                {
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )

    return tool_uses


def extract_thinking_content(message: AssistantMessage) -> str:
    """Extract thinking block content from message."""
    if not hasattr(message, "content"):
        return ""

    from claude_agent_sdk import ThinkingBlock

    thinking_parts = []
    for block in message.content:
        if isinstance(block, ThinkingBlock):
            thinking_parts.append(block.text)

    return "\n".join(thinking_parts)


def has_thinking_block(message: AssistantMessage) -> bool:
    """Check if message contains thinking block."""
    if not hasattr(message, "content"):
        return False

    from claude_agent_sdk import ThinkingBlock

    for block in message.content:
        if isinstance(block, ThinkingBlock):
            return True

    return False
