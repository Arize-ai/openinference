import json
from typing import Any

from openinference.instrumentation.google_genai._context import (
    CapturedRequestScope,
    _CapturedRequestWrapper,
    get_tool_attributes,
)
from openinference.semconv.trace import SpanAttributes, ToolAttributes


def test_get_tool_attributes_uses_parameters_json_schema() -> None:
    parameters_json_schema = {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    }
    request = {
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "Get the weather for a location",
                        "parametersJsonSchema": parameters_json_schema,
                    }
                ]
            }
        ]
    }

    def wrapped(*args: Any, **kwargs: Any) -> None:
        return None

    with CapturedRequestScope():
        _CapturedRequestWrapper()(
            wrapped,
            None,
            ("POST", "/models/gemini:generateContent", request),
            {},
        )
        attributes = dict(get_tool_attributes())

    tool_schema_key = f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}"
    tool_schema_json = attributes[tool_schema_key]
    assert isinstance(tool_schema_json, str)
    assert json.loads(tool_schema_json) == {
        "name": "get_weather",
        "description": "Get the weather for a location",
        "parameters": parameters_json_schema,
    }
