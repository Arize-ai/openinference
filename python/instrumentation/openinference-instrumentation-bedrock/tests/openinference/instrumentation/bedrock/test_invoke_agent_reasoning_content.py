"""Unit tests for Bedrock Agents (``invoke_agent``) reasoning content handling.

The agent runtime's ``modelInvocationOutput.rawResponse.content`` carries the
underlying model's raw Anthropic Messages API response, where extended-thinking
blocks use ``type: "thinking"`` / ``"redacted_thinking"`` (distinct from Bedrock
Converse's ``reasoningContent`` shape used by ``test_converse_reasoning_content.py``).
"""

from typing import Any, Dict

from openinference.instrumentation.bedrock._attribute_extractor import AttributeExtractor


def test_get_attributes_from_message_visible_thinking() -> None:
    message: Dict[str, Any] = {
        "type": "thinking",
        "thinking": "Let me work through this...",
        "signature": "sig-123",
    }
    result = AttributeExtractor.get_attributes_from_message(message, "assistant")
    assert result is not None
    assert result["role"] == "assistant"
    assert result["contents"] == [
        {
            "type": "reasoning",
            "text": "Let me work through this...",
            "signature": "sig-123",
        }
    ]


def test_get_attributes_from_message_redacted_thinking() -> None:
    message: Dict[str, Any] = {"type": "redacted_thinking", "data": "redacted-payload"}
    result = AttributeExtractor.get_attributes_from_message(message, "assistant")
    assert result is not None
    assert result["role"] == "assistant"
    assert result["contents"] == [{"type": "reasoning", "data": "redacted-payload"}]


def test_get_output_messages_preserves_order_with_thinking_and_text() -> None:
    model_output: Dict[str, Any] = {
        "rawResponse": {
            "content": (
                '{"content": ['
                '{"type": "thinking", "thinking": "Reasoning...", "signature": "sig-abc"}, '
                '{"type": "text", "text": "Final answer."}'
                "]}"
            )
        }
    }
    messages = AttributeExtractor.get_output_messages(model_output)  # type: ignore[arg-type]
    assert len(messages) == 2
    assert messages[0]["contents"] == [
        {"type": "reasoning", "text": "Reasoning...", "signature": "sig-abc"}
    ]
    assert messages[1]["content"] == "Final answer."


def test_get_output_messages_redacted_thinking() -> None:
    model_output: Dict[str, Any] = {
        "rawResponse": {
            "content": (
                '{"content": ['
                '{"type": "redacted_thinking", "data": "redacted-payload"}, '
                '{"type": "text", "text": "Final answer."}'
                "]}"
            )
        }
    }
    messages = AttributeExtractor.get_output_messages(model_output)  # type: ignore[arg-type]
    assert len(messages) == 2
    assert messages[0]["contents"] == [{"type": "reasoning", "data": "redacted-payload"}]
    assert messages[1]["content"] == "Final answer."
