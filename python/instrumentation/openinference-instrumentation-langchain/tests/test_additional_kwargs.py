import json

from openinference.instrumentation.langchain._tracer import _parse_message_data


def test_function_call_name() -> None:
    message_data = {
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "additional_kwargs": {
                "function_call": {
                    "name": "get_weather",
                    "arguments": {"city": "New York", "units": "metric"},  # Dict instead of string
                }
            }
        },
    }

    result = dict(_parse_message_data(message_data))
    assert "message.function_call_name" in result
    assert result["message.function_call_name"] == "get_weather"


def test_function_call_with_dict_arguments() -> None:
    """Test that function_call with dict arguments doesn't crash and serializes correctly"""
    message_data = {
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "additional_kwargs": {
                "function_call": {
                    "name": "get_weather",
                    "arguments": {"city": "New York", "units": "metric"},  # Dict instead of string
                }
            }
        },
    }

    result = dict(_parse_message_data(message_data))
    assert "message.function_call_arguments_json" in result
    parsed_args = json.loads(result["message.function_call_arguments_json"])
    assert parsed_args == {"city": "New York", "units": "metric"}


def test_function_call_with_string_arguments() -> None:
    """Test that function_call with string arguments still works (backward compatibility)"""
    message_data = {
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "additional_kwargs": {
                "function_call": {
                    "name": "get_weather",
                    "arguments": '{"city": "New York", "units": "metric"}',  # Correct string format
                }
            }
        },
    }

    result = dict(_parse_message_data(message_data))
    assert (
        result["message.function_call_arguments_json"] == '{"city": "New York", "units": "metric"}'
    )


def test_tool_call_name() -> None:
    """Test that tool_calls with dict arguments doesn't crash and serializes correctly"""
    message_data = {
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "additional_kwargs": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculator",
                            "arguments": {
                                "expression": "2+2",
                                "format": "int",
                            },  # Dict instead of string
                        }
                    }
                ]
            }
        },
    }

    result = dict(_parse_message_data(message_data))

    assert "message.tool_calls" in result
    tool_calls = result["message.tool_calls"]
    assert tool_calls[0]["tool_call.function.name"] == "calculator"


def test_tool_calls_with_dict_arguments() -> None:
    """Test that tool_calls with dict arguments doesn't crash and serializes correctly"""
    message_data = {
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "additional_kwargs": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculator",
                            "arguments": {
                                "expression": "2+2",
                                "format": "int",
                            },  # Dict instead of string
                        }
                    }
                ]
            }
        },
    }

    result = dict(_parse_message_data(message_data))

    tool_calls = result["message.tool_calls"]
    parsed_args = json.loads(tool_calls[0]["tool_call.function.arguments"])
    assert parsed_args == {"expression": "2+2", "format": "int"}


def test_tool_calls_with_string_arguments() -> None:
    """Test that tool_calls with string arguments still works (backward compatibility)"""
    message_data = {
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "additional_kwargs": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "2+2", "format": "int"}',
                        }
                    }
                ]
            }
        },
    }

    result = dict(_parse_message_data(message_data))

    tool_calls = result["message.tool_calls"]
    assert tool_calls[0]["tool_call.function.arguments"] == '{"expression": "2+2", "format": "int"}'


########################################################
# Complex nested arguments
########################################################


def test_complex_nested_arguments_serialization() -> None:
    """Test that complex nested objects in arguments are properly serialized"""
    message_data = {
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "additional_kwargs": {
                "function_call": {
                    "name": "complex_tool",
                    "arguments": {
                        "user": {"id": 123, "name": "John"},
                        "request": {
                            "timestamp": "2024-01-01",
                            "params": {"filter": ["active", "verified"], "limit": 10},
                        },
                    },
                }
            }
        },
    }

    result = dict(_parse_message_data(message_data))

    parsed_args = json.loads(result["message.function_call_arguments_json"])
    expected = {
        "user": {"id": 123, "name": "John"},
        "request": {
            "timestamp": "2024-01-01",
            "params": {"filter": ["active", "verified"], "limit": 10},
        },
    }
    assert parsed_args == expected


def test_mixed_function_call_and_tool_calls_with_dict_args() -> None:
    """Test handling both function_call and tool_calls with dict arguments"""
    message_data = {
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "additional_kwargs": {
                "function_call": {
                    "name": "primary_tool",
                    "arguments": {"action": "search", "query": "test"},
                },
                "tool_calls": [
                    {
                        "function": {
                            "name": "secondary_tool",
                            "arguments": {"validation": True, "retry": 3},
                        }
                    }
                ],
            }
        },
    }

    result = dict(_parse_message_data(message_data))

    func_args = json.loads(result["message.function_call_arguments_json"])
    tool_args = json.loads(result["message.tool_calls"][0]["tool_call.function.arguments"])

    assert func_args == {"action": "search", "query": "test"}
    assert tool_args == {"validation": True, "retry": 3}


# ########################################################
# # Edge case tests for robustness
# ########################################################


def test_non_serializable_arguments_fallback() -> None:
    """Test handling of non-JSON-serializable arguments"""

    class NonSerializable:
        def __init__(self) -> None:
            self.data = "test"

    non_serializable = NonSerializable()
    message_data = {
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "additional_kwargs": {
                "function_call": {"name": "test_tool", "arguments": {"obj": non_serializable}}
            }
        },
    }

    result = dict(_parse_message_data(message_data))
    assert "message.function_call_arguments_json" in result


def test_very_large_arguments_performance() -> None:
    """Test performance with large argument objects"""
    large_args = {
        "data": ["item_{}".format(i) for i in range(1000)],
        "metadata": {"key_{}".format(i): f"value_{i}" for i in range(100)},
    }

    message_data = {
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "additional_kwargs": {
                "function_call": {"name": "large_data_tool", "arguments": large_args}
            }
        },
    }

    # Should not crash or timeout
    result = dict(_parse_message_data(message_data))

    assert "message.function_call_arguments_json" in result
    parsed_args = json.loads(result["message.function_call_arguments_json"])
    assert len(parsed_args["data"]) == 1000
    assert len(parsed_args["metadata"]) == 100
