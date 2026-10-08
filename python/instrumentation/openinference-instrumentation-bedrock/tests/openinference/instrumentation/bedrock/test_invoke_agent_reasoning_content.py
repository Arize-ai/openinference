"""Unit tests for Bedrock Agents (``invoke_agent``) reasoning content handling.

The agent runtime's ``modelInvocationOutput.rawResponse.content`` carries the underlying
model's raw response, which can appear in two different shapes regardless of the model
provider (Anthropic and non-Anthropic models have both been observed producing either
shape in live traces):

- Provider-native (e.g. Anthropic Messages API): blocks are tagged with a ``type`` field,
  and extended-thinking blocks use ``type: "thinking"`` / ``"redacted_thinking"``.
- Bedrock Converse-normalized (``output.message.content``): blocks have no ``type``
  discriminator; every field is present explicitly, with unset fields set to ``null``
  rather than omitted, so fields must be checked for truthiness rather than key presence.

Separately, ``modelInvocationOutput`` may also carry a structured, top-level
``reasoningContent`` field (sibling to ``rawResponse``); when present it is captured first
and any duplicate reasoning block inside ``rawResponse`` is skipped.
"""

import json
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
    assert "id" not in result["contents"][0]


def test_get_attributes_from_message_redacted_thinking() -> None:
    message: Dict[str, Any] = {"type": "redacted_thinking", "data": "redacted-payload"}
    result = AttributeExtractor.get_attributes_from_message(message, "assistant")
    assert result is not None
    assert result["role"] == "assistant"
    assert result["contents"] == [{"type": "reasoning", "data": "redacted-payload"}]
    assert "id" not in result["contents"][0]


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
    assert len(messages) == 1
    assert messages[0]["contents"] == [
        {"type": "reasoning", "text": "Reasoning...", "signature": "sig-abc"},
        {"type": "text", "text": "Final answer."},
    ]
    assert "id" not in messages[0]["contents"][0]


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
    assert len(messages) == 1
    assert messages[0]["contents"] == [
        {"type": "reasoning", "data": "redacted-payload"},
        {"type": "text", "text": "Final answer."},
    ]
    assert "id" not in messages[0]["contents"][0]


def test_get_attributes_from_message_converse_text_block() -> None:
    message: Dict[str, Any] = {
        "text": "The 10th Fibonacci number is 55.",
        "image": None,
        "reasoningContent": None,
        "toolUse": None,
    }
    result = AttributeExtractor.get_attributes_from_message(message, "assistant")
    assert result == {"content": "The 10th Fibonacci number is 55.", "role": "assistant"}


def test_get_attributes_from_message_converse_reasoning_block() -> None:
    message: Dict[str, Any] = {
        "text": None,
        "reasoningContent": {
            "reasoningText": {"text": "Let's compute step by step.", "signature": None},
            "redactedContent": None,
        },
        "toolUse": None,
    }
    result = AttributeExtractor.get_attributes_from_message(message, "assistant")
    assert result is not None
    assert result["role"] == "assistant"
    assert result["contents"] == [{"type": "reasoning", "text": "Let's compute step by step."}]
    assert "id" not in result["contents"][0]


def test_get_attributes_from_message_converse_tool_use_block() -> None:
    message: Dict[str, Any] = {
        "text": None,
        "reasoningContent": None,
        "toolUse": {
            "toolUseId": "tooluse_1G5NXAHjWrWdbstFwz5PEF",
            "name": "prime_numbers_between_n1_and_n2",
            "input": {"n1": 10, "n2": 50},
        },
    }
    result = AttributeExtractor.get_attributes_from_message(message, "assistant")
    assert result is not None
    assert result["role"] == "tool"
    assert result["tool_call_id"] == "tooluse_1G5NXAHjWrWdbstFwz5PEF"
    assert result["tool_calls"] == [
        {
            "id": "tooluse_1G5NXAHjWrWdbstFwz5PEF",
            "function": {
                "name": "prime_numbers_between_n1_and_n2",
                "arguments": {"n1": 10, "n2": 50},
            },
        }
    ]


def test_get_attributes_from_message_converse_all_null_returns_none() -> None:
    message: Dict[str, Any] = {
        "text": None,
        "image": None,
        "reasoningContent": None,
        "toolUse": None,
        "toolResult": None,
    }
    assert AttributeExtractor.get_attributes_from_message(message, "assistant") is None


def test_get_output_messages_converse_style_raw_response_without_structured_field() -> None:
    """Anthropic model orchestration output with no extended thinking enabled: a Converse-
    normalized rawResponse with a single plain text block and no top-level reasoningContent
    (matches a live anthropic.claude-sonnet-4-6 agent trace)."""
    raw_content = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "text": "The 10th Fibonacci number is **55**.",
                        "reasoningContent": None,
                    }
                ],
            }
        },
        "stopReason": "stop_sequence",
    }
    model_output: Dict[str, Any] = {"rawResponse": {"content": json.dumps(raw_content)}}
    messages = AttributeExtractor.get_output_messages(model_output)  # type: ignore[arg-type]
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == "The 10th Fibonacci number is **55**."


def test_get_output_messages_converse_style_raw_response_gpt_oss_dedup() -> None:
    """GPT-OSS orchestration output: the structured top-level reasoningContent field
    duplicates the reasoning block inside the Converse-normalized rawResponse, so only the
    final answer text block should additionally be captured (matches a live
    openai.gpt-oss-120b-1:0 inline-agent trace)."""
    raw_content = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "text": None,
                        "reasoningContent": {
                            "reasoningText": {
                                "text": "We need to compute the 10th Fibonacci number...",
                                "signature": None,
                            },
                            "redactedContent": None,
                        },
                    },
                    {
                        "text": "<answer>The 10th Fibonacci number is 55.</answer>",
                        "reasoningContent": None,
                    },
                ],
            }
        },
        "stopReason": "end_turn",
    }
    model_output: Dict[str, Any] = {
        "rawResponse": {"content": json.dumps(raw_content)},
        "reasoningContent": {
            "reasoningText": {"text": "We need to compute the 10th Fibonacci number..."}
        },
    }
    messages = AttributeExtractor.get_output_messages(model_output)  # type: ignore[arg-type]
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert messages[0]["contents"][0]["type"] == "reasoning"
    assert messages[0]["contents"][1] == {
        "type": "text",
        "text": "<answer>The 10th Fibonacci number is 55.</answer>",
    }


def test_get_output_messages_converse_style_raw_response_with_tool_use() -> None:
    """Orchestration step that reasons then calls a tool, Converse-normalized rawResponse
    (matches the toolUse block seen in test_routing_classifier_with_reasoning's cassette)."""
    raw_content = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "text": None,
                        "reasoningContent": {
                            "reasoningText": {"text": "Need to call the tool.", "signature": None},
                            "redactedContent": None,
                        },
                    },
                    {
                        "text": None,
                        "toolUse": {
                            "toolUseId": "tooluse_abc",
                            "name": "prime_numbers_between_n1_and_n2",
                            "input": {"n1": 10, "n2": 50},
                        },
                    },
                ],
            }
        },
        "stopReason": "tool_use",
    }
    model_output: Dict[str, Any] = {"rawResponse": {"content": json.dumps(raw_content)}}
    messages = AttributeExtractor.get_output_messages(model_output)  # type: ignore[arg-type]
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert messages[0]["contents"][0]["type"] == "reasoning"
    assert "content" not in messages[0]
    assert messages[0]["tool_calls"][0]["function"]["name"] == ("prime_numbers_between_n1_and_n2")
