"""Tests for LangGraph tool call instrumentation to ensure tool calls appear in output traces."""

import json

from openinference.instrumentation.langchain._tracer import (
    MESSAGE_CONTENT,
    MESSAGE_ROLE,
    MESSAGE_TOOL_CALLS,
    TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
    TOOL_CALL_FUNCTION_NAME,
    TOOL_CALL_ID,
    _parse_message_data,
)


def test_parse_message_data_with_langgraph_tool_calls() -> None:
    """Test _parse_message_data extracts tool calls from LangGraph-style message data."""
    message_data = {
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "content": "I'll help you with both requests.",
            "additional_kwargs": {},
            "tool_calls": [
                {"name": "get_weather", "args": {"city": "San Francisco"}, "id": "call_1"},
                {"name": "calculate_sum", "args": {"a": 5, "b": 3}, "id": "call_2"},
            ],
        },
    }

    # Parse the message data
    parsed_attributes = dict(_parse_message_data(message_data))

    # Verify role is extracted
    assert parsed_attributes[MESSAGE_ROLE] == "assistant"

    # Verify content is extracted
    assert parsed_attributes[MESSAGE_CONTENT] == "I'll help you with both requests."

    # Verify tool calls are extracted
    assert MESSAGE_TOOL_CALLS in parsed_attributes
    tool_calls = parsed_attributes[MESSAGE_TOOL_CALLS]

    assert len(tool_calls) == 2

    # Verify first tool call
    weather_call = next(tc for tc in tool_calls if tc.get(TOOL_CALL_FUNCTION_NAME) == "get_weather")
    assert weather_call[TOOL_CALL_ID] == "call_1"
    assert json.loads(weather_call[TOOL_CALL_FUNCTION_ARGUMENTS_JSON]) == {"city": "San Francisco"}

    # Verify second tool call
    sum_call = next(tc for tc in tool_calls if tc.get(TOOL_CALL_FUNCTION_NAME) == "calculate_sum")
    assert sum_call[TOOL_CALL_ID] == "call_2"
    assert json.loads(sum_call[TOOL_CALL_FUNCTION_ARGUMENTS_JSON]) == {"a": 5, "b": 3}
