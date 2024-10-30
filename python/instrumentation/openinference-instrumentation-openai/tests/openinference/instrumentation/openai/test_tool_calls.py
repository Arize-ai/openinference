import json
from importlib import import_module
from importlib.metadata import version
from typing import Tuple, cast

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_tool_calls(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
) -> None:
    if _openai_version() < (1, 12, 0):
        pytest.skip("Not supported")
    openai = import_module("openai")
    from openai.types.chat import (  # type: ignore[attr-defined,unused-ignore]
        ChatCompletionToolParam,
    )

    client = openai.OpenAI(api_key="sk-")
    input_tools = [
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "get_weather",
                "description": "finds the weather for a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for, e.g. 'London'",
                        }
                    },
                    "required": ["city"],
                },
            },
        ),
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "get_population",
                "description": "finds the population for a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the population for, e.g. 'London'",
                        }
                    },
                    "required": ["city"],
                },
            },
        ),
    ]
    client.chat.completions.create(
        extra_headers={"Accept-Encoding": "gzip"},
        model="gpt-4o-mini",
        tools=input_tools,
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_62136355",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "New York"}'},
                    },
                    {
                        "id": "call_62136356",
                        "type": "function",
                        "function": {"name": "get_population", "arguments": '{"city": "New York"}'},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_62136355",
                "content": '{"city": "New York", "weather": "fine"}',
            },
            {
                "role": "tool",
                "tool_call_id": "call_62136356",
                "content": '{"city": "New York", "weather": "large"}',
            },
            {
                "role": "assistant",
                "content": "In New York the weather is fine and the population is large.",
            },
            {
                "role": "user",
                "content": "What's the weather and population in San Francisco?",
            },
        ],
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})
    for i in range(len(input_tools)):
        json_schema = attributes.pop(f"llm.tools.{i}.tool.json_schema")
        assert isinstance(json_schema, str)
        assert json.loads(json_schema)
    assert (
        attributes.pop("llm.input_messages.0.message.tool_calls.0.tool_call.id") == "call_62136355"
    )
    assert (
        attributes.pop("llm.input_messages.0.message.tool_calls.0.tool_call.function.name")
        == "get_weather"
    )
    assert (
        attributes.pop("llm.input_messages.0.message.tool_calls.0.tool_call.function.arguments")
        == '{"city": "New York"}'
    )
    assert (
        attributes.pop("llm.input_messages.0.message.tool_calls.1.tool_call.id") == "call_62136356"
    )
    assert (
        attributes.pop("llm.input_messages.0.message.tool_calls.1.tool_call.function.name")
        == "get_population"
    )
    assert (
        attributes.pop("llm.input_messages.0.message.tool_calls.1.tool_call.function.arguments")
        == '{"city": "New York"}'
    )
    assert attributes.pop("llm.input_messages.1.message.role") == "tool"
    assert attributes.pop("llm.input_messages.1.message.tool_call_id") == "call_62136355"
    assert (
        attributes.pop("llm.input_messages.1.message.content")
        == '{"city": "New York", "weather": "fine"}'
    )
    assert attributes.pop("llm.input_messages.2.message.role") == "tool"
    assert attributes.pop("llm.input_messages.2.message.tool_call_id") == "call_62136356"
    assert (
        attributes.pop("llm.input_messages.2.message.content")
        == '{"city": "New York", "weather": "large"}'
    )
    assert attributes.pop("llm.output_messages.0.message.tool_calls.0.tool_call.id")
    assert (
        attributes.pop("llm.output_messages.0.message.tool_calls.0.tool_call.function.name")
        == "get_weather"
    )
    assert (
        attributes.pop("llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments")
        == '{"city": "San Francisco"}'
    )
    assert attributes.pop("llm.output_messages.0.message.tool_calls.1.tool_call.id")
    assert (
        attributes.pop("llm.output_messages.0.message.tool_calls.1.tool_call.function.name")
        == "get_population"
    )
    assert (
        attributes.pop("llm.output_messages.0.message.tool_calls.1.tool_call.function.arguments")
        == '{"city": "San Francisco"}'
    )


def _openai_version() -> Tuple[int, int, int]:
    return cast(Tuple[int, int, int], tuple(map(int, version("openai").split(".")[:3])))
