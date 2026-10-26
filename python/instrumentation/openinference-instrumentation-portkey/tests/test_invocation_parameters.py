import json
from inspect import signature
from typing import Any, Dict

from openinference.semconv.trace import SpanAttributes
from portkey_ai.api_resources.apis.chat_complete import Completions

from openinference.instrumentation.portkey._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.instrumentation.portkey._wrappers import _parse_args

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]


def _extra_attributes(request_parameters: Dict[str, Any]) -> Dict[str, Any]:
    extractor = _RequestAttributesExtractor()
    return dict(extractor.get_extra_attributes_from_request(request_parameters))


def test_parse_args_records_only_caller_supplied_parameters() -> None:
    request_parameters = _parse_args(
        signature(Completions.create),
        None,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "weather?"}],
    )

    assert set(request_parameters) == {"model", "messages"}

    invocation_parameters = json.loads(
        _extra_attributes(request_parameters)[SpanAttributes.LLM_INVOCATION_PARAMETERS]
    )
    assert invocation_parameters == {"model": "gpt-4o-mini"}


def test_parse_args_surfaces_tools_passed_through_kwargs() -> None:
    request_parameters = _parse_args(
        signature(Completions.create),
        None,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "weather?"}],
        tools=TOOLS,
        tool_choice="auto",
    )

    assert "kwargs" not in request_parameters
    assert request_parameters["tools"] == TOOLS
    assert request_parameters["tool_choice"] == "auto"

    attributes = _extra_attributes(request_parameters)
    assert attributes["llm.tools.0.tool.json_schema"] == json.dumps(TOOLS[0])

    invocation_parameters = json.loads(attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS])
    assert invocation_parameters == {"model": "gpt-4o-mini", "tool_choice": "auto"}
    assert all(
        "Omit" not in str(value) and "NotGiven" not in str(value)
        for value in invocation_parameters.values()
    )
