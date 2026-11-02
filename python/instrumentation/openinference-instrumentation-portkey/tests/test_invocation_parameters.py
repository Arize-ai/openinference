import json
import os
from inspect import signature
from typing import Any, Dict

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from portkey_ai import AsyncPortkey, Portkey
from portkey_ai.api_resources.apis.chat_complete import Completions

from openinference.instrumentation.portkey._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.instrumentation.portkey._wrappers import _parse_args
from openinference.semconv.trace import SpanAttributes

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

_SAFE_REQUEST_HEADERS = frozenset(
    {"accept", "accept-encoding", "content-type", "host", "user-agent"}
)
_SAFE_RESPONSE_HEADERS = frozenset({"content-type", "date", "server", "cf-cache-status"})


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


@pytest.mark.parametrize("sentinel_name", ["sdk_default", "NotGiven", "Omit"])
def test_parse_args_omits_explicit_sdk_sentinels(sentinel_name: str) -> None:
    create_signature = signature(Completions.create)
    sentinel = (
        create_signature.parameters["temperature"].default
        if sentinel_name == "sdk_default"
        else type(sentinel_name, (), {})()
    )
    request_parameters = _parse_args(
        create_signature,
        None,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "weather?"}],
        temperature=sentinel,
        tool_choice=sentinel,
    )

    assert request_parameters == {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "weather?"}],
    }


def test_parse_args_does_not_add_portkey_default_model() -> None:
    request_parameters = _parse_args(
        signature(Completions.create),
        None,
        messages=[{"role": "user", "content": "weather?"}],
    )

    assert request_parameters == {"messages": [{"role": "user", "content": "weather?"}]}


def _assert_tool_span(exporter: InMemorySpanExporter) -> None:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    assert attributes["llm.tools.0.tool.json_schema"] == json.dumps(TOOLS[0])
    invocation_parameters = attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS]
    assert isinstance(invocation_parameters, str)
    assert json.loads(invocation_parameters) == {
        "model": "gpt-4o-mini",
        "tool_choice": "auto",
    }
    input_value = attributes[SpanAttributes.INPUT_VALUE]
    assert isinstance(input_value, str)
    assert json.loads(input_value) == {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": TOOLS,
        "tool_choice": "auto",
    }


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda request: setattr(  # type: ignore[func-returns-value]
        request,
        "headers",
        {
            k: v
            for k, v in request.headers.items()
            if k.lower() in _SAFE_REQUEST_HEADERS or k.lower().startswith("x-stainless-")
        },
    )
    or request,
    before_record_response=lambda response: {
        **response,
        "headers": {
            k: v
            for k, v in response["headers"].items()
            if k.lower() in _SAFE_RESPONSE_HEADERS or k.lower().startswith("x-ratelimit-")
        },
    },
)
def test_chat_completion_records_tools(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_portkey_instrumentation: None,
    vcr_cassette: Any,
) -> None:
    Portkey(
        api_key=os.getenv("PORTKEY_API_KEY", "REDACTED"),
        virtual_key=os.getenv("PORTKEY_VIRTUAL_KEY", "REDACTED"),
    ).chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "weather?"}],
        tools=TOOLS,
        tool_choice="auto",
    )

    assert vcr_cassette.requests
    assert json.loads(vcr_cassette.requests[-1].body)["tools"] == TOOLS
    _assert_tool_span(in_memory_span_exporter)


@pytest.mark.asyncio
@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda request: setattr(  # type: ignore[func-returns-value]
        request,
        "headers",
        {
            k: v
            for k, v in request.headers.items()
            if k.lower() in _SAFE_REQUEST_HEADERS or k.lower().startswith("x-stainless-")
        },
    )
    or request,
    before_record_response=lambda response: {
        **response,
        "headers": {
            k: v
            for k, v in response["headers"].items()
            if k.lower() in _SAFE_RESPONSE_HEADERS or k.lower().startswith("x-ratelimit-")
        },
    },
)
async def test_async_chat_completion_records_tools(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_portkey_instrumentation: None,
    vcr_cassette: Any,
) -> None:
    await AsyncPortkey(
        api_key=os.getenv("PORTKEY_API_KEY", "REDACTED"),
        virtual_key=os.getenv("PORTKEY_VIRTUAL_KEY", "REDACTED"),
    ).chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "weather?"}],
        tools=TOOLS,
        tool_choice="auto",
    )

    assert vcr_cassette.requests
    assert json.loads(vcr_cassette.requests[-1].body)["tools"] == TOOLS
    _assert_tool_span(in_memory_span_exporter)
