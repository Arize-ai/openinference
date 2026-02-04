import pytest
from claude_agent_sdk import AssistantMessage, TextBlock, ThinkingBlock, ToolUseBlock

from openinference.instrumentation.claude_code._message_parser import (
    extract_text_content,
    extract_thinking_content,
    extract_tool_uses,
    has_thinking_block,
)


def test_extract_text_content_from_assistant_message():
    """Test extracting text from AssistantMessage."""
    message = AssistantMessage(
        content=[
            TextBlock(type="text", text="Hello, world!"),
            TextBlock(type="text", text="How are you?"),
        ]
    )

    result = extract_text_content(message)

    assert result == "Hello, world!\nHow are you?"


def test_extract_text_content_from_empty_message():
    """Test extracting text from message with no text blocks."""
    message = AssistantMessage(content=[])

    result = extract_text_content(message)

    assert result == ""


def test_extract_tool_uses_from_message():
    """Test extracting tool use blocks."""
    message = AssistantMessage(
        content=[
            TextBlock(type="text", text="I'll read the file"),
            ToolUseBlock(
                type="tool_use",
                id="tool_123",
                name="Read",
                input={"file_path": "test.py"}
            ),
        ]
    )

    result = extract_tool_uses(message)

    assert len(result) == 1
    assert result[0]["id"] == "tool_123"
    assert result[0]["name"] == "Read"
    assert result[0]["input"]["file_path"] == "test.py"


def test_extract_thinking_content():
    """Test extracting thinking block content."""
    message = AssistantMessage(
        content=[
            ThinkingBlock(type="thinking", text="Let me think about this..."),
            TextBlock(type="text", text="The answer is 4"),
        ]
    )

    result = extract_thinking_content(message)

    assert result == "Let me think about this..."


def test_has_thinking_block_returns_true():
    """Test detecting thinking block presence."""
    message = AssistantMessage(
        content=[
            ThinkingBlock(type="thinking", text="Thinking..."),
        ]
    )

    result = has_thinking_block(message)

    assert result is True


def test_has_thinking_block_returns_false():
    """Test detecting no thinking block."""
    message = AssistantMessage(
        content=[
            TextBlock(type="text", text="Answer"),
        ]
    )

    result = has_thinking_block(message)

    assert result is False
