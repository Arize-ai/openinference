import json
import random
import string
from importlib import import_module
from importlib.metadata import version
from typing import Tuple, cast

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.semconv.trace import OpenInferenceLLMProviderValues, SpanAttributes


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
    llm_spans = get_openai_llm_spans(spans, 1)
    assert len(llm_spans) == 1
    span = llm_spans[0]
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


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_cached_tokens(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
) -> None:
    if _openai_version() < (1, 12, 0):
        pytest.skip("Not supported")
    openai = import_module("openai")

    client = openai.OpenAI(api_key="sk-")
    random_1024_token_prefix = "".join(random.choices(string.ascii_letters + string.digits, k=2000))
    client.chat.completions.create(
        extra_headers={"Accept-Encoding": "gzip"},
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "{} Write me a haiku.".format(random_1024_token_prefix),
            },
        ],
    )
    client.chat.completions.create(
        extra_headers={"Accept-Encoding": "gzip"},
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "{} Write me a sonnet.".format(random_1024_token_prefix),
            },
        ],
    )
    spans = in_memory_span_exporter.get_finished_spans()
    llm_spans = get_openai_llm_spans(spans, 2)
    assert len(llm_spans) == 2
    span = llm_spans[1]
    attributes = dict(span.attributes or {})
    assert attributes.pop("llm.token_count.prompt_details.cache_read") == 1280


def _openai_version() -> Tuple[int, int, int]:
    return cast(Tuple[int, int, int], tuple(map(int, version("openai").split(".")[:3])))


def get_openai_llm_spans(
    spans: Tuple[ReadableSpan], fallback_count: int = 1
) -> Tuple[ReadableSpan]:
    """Return all OpenAI LLM spans, fallback to OpenAI TOOL spans if needed."""
    # Find the primary OpenAI response span (v1: one span, v2: one of many)
    llm_spans = [
        span
        for span in spans
        if span.attributes
        and span.attributes.get(SpanAttributes.LLM_SYSTEM)
        == OpenInferenceLLMProviderValues.OPENAI.value
    ]
    # Fallback to pick the span that actually has tool attributes
    if len(llm_spans) != fallback_count:
        llm_spans = [
            span
            for span in spans
            if span.attributes and any(k.startswith("llm.tools.") for k in span.attributes)
        ]
    if not llm_spans:
        raise ValueError("No OpenAI LLM spans found in spans.")
    return tuple(llm_spans)
