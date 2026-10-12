"""Unit tests for the GenAI to OpenInference attribute mapping.

The attribute payloads mirror what agent-framework-core 1.19 emits for a
tool-calling agent on the OpenAI Responses API, where each chat span only
carries the new turn and the system prompt is in gen_ai.system_instructions.
"""

import json
from typing import Any, Dict, List

import pytest

from openinference.instrumentation.agent_framework.semantic_conventions import get_attributes

SYSTEM_INSTRUCTIONS = json.dumps([{"type": "text", "content": "You are a weather assistant."}])
USER_MESSAGES = json.dumps(
    [{"role": "user", "parts": [{"type": "text", "content": "What's the weather in Amsterdam?"}]}]
)
TOOL_CALL_OUTPUT = json.dumps(
    [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"location":"Amsterdam"}',
                }
            ],
            "finish_reason": "tool_call",
        }
    ]
)


def _attributes(attrs: Dict[str, Any], span_name: str) -> Dict[str, Any]:
    return dict(get_attributes(attrs, span_name, 1234))


def _chat_attrs(**overrides: Any) -> Dict[str, Any]:
    attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-5.5",
        "gen_ai.system_instructions": SYSTEM_INSTRUCTIONS,
        "gen_ai.input.messages": USER_MESSAGES,
        "gen_ai.output.messages": TOOL_CALL_OUTPUT,
        "gen_ai.usage.input_tokens": 80,
        "gen_ai.usage.output_tokens": 18,
    }
    attrs.update(overrides)
    return attrs


def test_input_messages_are_only_emitted_flattened() -> None:
    result = _attributes(_chat_attrs(), "chat gpt-5.5")

    assert "llm.input_messages" not in result
    assert "llm.output_messages" not in result
    assert "llm.input_messages.2.message.role" not in result


def test_system_instructions_become_first_input_message() -> None:
    result = _attributes(_chat_attrs(), "chat gpt-5.5")

    assert result["llm.input_messages.0.message.role"] == "system"
    assert result["llm.input_messages.0.message.content"] == "You are a weather assistant."
    assert result["llm.input_messages.1.message.role"] == "user"
    assert result["llm.input_messages.1.message.content"] == "What's the weather in Amsterdam?"


def test_system_instructions_do_not_change_plain_text_input_value() -> None:
    result = _attributes(_chat_attrs(), "chat gpt-5.5")

    assert result["input.value"] == "What's the weather in Amsterdam?"
    assert result["input.mime_type"] == "text/plain"


def test_system_instructions_as_plain_string() -> None:
    result = _attributes(_chat_attrs(**{"gen_ai.system_instructions": "Be brief."}), "chat gpt-5.5")

    assert result["llm.input_messages.0.message.role"] == "system"
    assert result["llm.input_messages.0.message.content"] == "Be brief."


def test_system_prompt_already_in_input_messages_is_not_duplicated() -> None:
    # Agent Framework 1.0 prepends the instructions to gen_ai.input.messages
    # and also records them in gen_ai.system_instructions
    input_messages = json.dumps(
        [
            {
                "role": "system",
                "parts": [{"type": "text", "content": "You are a weather assistant."}],
            },
            json.loads(USER_MESSAGES)[0],
        ]
    )
    result = _attributes(_chat_attrs(**{"gen_ai.input.messages": input_messages}), "chat gpt-5.5")

    assert result["llm.input_messages.0.message.role"] == "system"
    assert result["llm.input_messages.1.message.role"] == "user"
    assert "llm.input_messages.2.message.role" not in result
    assert result["input.value"] == "What's the weather in Amsterdam?"
    assert result["input.mime_type"] == "text/plain"


def test_multiple_system_instructions_become_separate_messages() -> None:
    instructions = json.dumps(
        [{"type": "text", "content": "Be brief."}, {"type": "text", "content": "Use Celsius."}]
    )
    result = _attributes(
        _chat_attrs(**{"gen_ai.system_instructions": instructions}), "chat gpt-5.5"
    )

    assert result["llm.input_messages.0.message.content"] == "Be brief."
    assert result["llm.input_messages.1.message.content"] == "Use Celsius."
    assert result["llm.input_messages.2.message.role"] == "user"


@pytest.mark.parametrize(
    "instructions, expected",
    [
        pytest.param("42", ["42"], id="text-that-parses-as-json-number"),
        pytest.param('"Be brief."', ['"Be brief."'], id="text-that-parses-as-json-string"),
        pytest.param(["Be brief.", "Use Celsius."], ["Be brief.", "Use Celsius."], id="list"),
        pytest.param(("Be brief.", "Use Celsius."), ["Be brief.", "Use Celsius."], id="tuple"),
    ],
)
def test_system_instructions_in_other_shapes(instructions: Any, expected: List[str]) -> None:
    result = _attributes(
        _chat_attrs(**{"gen_ai.system_instructions": instructions}), "chat gpt-5.5"
    )

    contents = [result[f"llm.input_messages.{i}.message.content"] for i in range(len(expected))]
    assert contents == expected
    assert result[f"llm.input_messages.{len(expected)}.message.role"] == "user"


def test_json_input_value_includes_system_prompt() -> None:
    tool_result = json.dumps(
        [
            {
                "role": "tool",
                "parts": [
                    {"type": "tool_call_response", "id": "call_1", "response": "It is sunny."}
                ],
            }
        ]
    )
    result = _attributes(_chat_attrs(**{"gen_ai.input.messages": tool_result}), "chat gpt-5.5")

    assert result["input.mime_type"] == "application/json"
    roles = [m["message.role"] for m in json.loads(result["input.value"])["messages"]]
    assert roles == ["system", "tool"]


def test_tool_call_arguments_are_not_double_encoded() -> None:
    result = _attributes(_chat_attrs(), "chat gpt-5.5")

    arguments = result["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"]
    assert json.loads(arguments) == {"location": "Amsterdam"}


def test_tool_call_arguments_dict_is_serialized() -> None:
    output = json.dumps(
        [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "id": "call_1",
                        "name": "get_weather",
                        "arguments": {"location": "Amsterdam"},
                    }
                ],
            }
        ]
    )
    result = _attributes(_chat_attrs(**{"gen_ai.output.messages": output}), "chat gpt-5.5")

    arguments = result["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"]
    assert json.loads(arguments) == {"location": "Amsterdam"}


def test_tool_calling_turn_output_value_includes_tool_calls() -> None:
    result = _attributes(_chat_attrs(), "chat gpt-5.5")

    message = json.loads(result["output.value"])["choices"][0]["message"]
    assert message["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"location":"Amsterdam"}'},
        }
    ]


def test_tool_result_turn_keeps_single_tool_message() -> None:
    tool_result = json.dumps(
        [
            {
                "role": "tool",
                "parts": [
                    {
                        "type": "tool_call_response",
                        "id": "call_1",
                        "response": "The weather in Amsterdam is sunny.",
                    }
                ],
            }
        ]
    )
    result = _attributes(_chat_attrs(**{"gen_ai.input.messages": tool_result}), "chat gpt-5.5")

    assert result["llm.input_messages.0.message.role"] == "system"
    assert result["llm.input_messages.1.message.role"] == "tool"
    assert result["llm.input_messages.1.message.tool_call_id"] == "call_1"
    assert result["llm.input_messages.1.message.content"] == "The weather in Amsterdam is sunny."
    assert "llm.input_messages.2.message.role" not in result


def test_agent_span_has_system_prompt_and_plain_text_io() -> None:
    output = json.dumps(
        [
            json.loads(TOOL_CALL_OUTPUT)[0],
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "It is sunny in Amsterdam."}],
            },
        ]
    )
    attrs = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "WeatherAgent",
        "gen_ai.system_instructions": SYSTEM_INSTRUCTIONS,
        "gen_ai.input.messages": USER_MESSAGES,
        "gen_ai.output.messages": output,
    }
    result = _attributes(attrs, "invoke_agent WeatherAgent")

    assert result["openinference.span.kind"] == "AGENT"
    assert result["llm.input_messages.0.message.role"] == "system"
    assert result["input.value"] == "What's the weather in Amsterdam?"
    assert result["output.value"] == "It is sunny in Amsterdam."
