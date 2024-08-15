import json
from contextlib import suppress
from importlib import import_module
from importlib.metadata import version
from typing import Tuple, cast

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.mark.disable_socket
def test_tool_call(
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
    with suppress(openai.APIConnectionError):
        client.chat.completions.create(
            model="gpt-4",
            tools=input_tools,
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather like in San Francisco?",
                },
            ],
        )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 4
    span = spans[3]
    attributes = span.attributes or dict()
    for i in range(len(input_tools)):
        json_schema = attributes.get(f"llm.tools.{i}.tool.json_schema")
        assert isinstance(json_schema, str)
        assert json.loads(json_schema)


def _openai_version() -> Tuple[int, int, int]:
    return cast(Tuple[int, int, int], tuple(map(int, version("openai").split(".")[:3])))
