import asyncio
import json
from typing import Any, Dict, List

import litellm
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.litellm import LiteLLMInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
resource = Resource.create(
    {
        "service.name": "litellm-tools-streaming-example",
        "openinference.project.name": "openinference-litellm-tools-streaming",
    }
)
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)


tools: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

messages: List[Dict[str, Any]] = [
    {
        "role": "user",
        "content": "What's the weather like in San Francisco, Tokyo, and Paris?",
    }
]


def run_sync_stream() -> None:
    print("\n=== sync streaming ===")
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=True,
    )
    chunk_count = 0
    for chunk in response:
        chunk_count += 1
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and getattr(delta, "tool_calls", None):
            for tc in delta.tool_calls:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None)
                args = getattr(fn, "arguments", None)
                print(f"  delta tool_call[{tc.index}] name={name!r} args={args!r}")
    print(f"received {chunk_count} chunks")


async def run_async_stream() -> None:
    print("\n=== async streaming ===")
    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=True,
    )
    chunk_count = 0
    async for chunk in response:
        chunk_count += 1
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and getattr(delta, "tool_calls", None):
            for tc in delta.tool_calls:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None)
                args = getattr(fn, "arguments", None)
                print(f"  delta tool_call[{tc.index}] name={name!r} args={args!r}")
    print(f"received {chunk_count} chunks")


if __name__ == "__main__":
    run_sync_stream()
    asyncio.run(run_async_stream())
    tracer_provider.force_flush()
    # Demonstrate the round-trip too: send the tool result back as a follow-up
    # streamed call so the second span shows assistant content streaming.
    print("\n=== sync streaming follow-up with tool result ===")
    follow_up_messages: List[Dict[str, Any]] = [
        *messages,
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_demo_1",
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "arguments": json.dumps({"location": "San Francisco, CA"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_demo_1",
            "name": "get_current_weather",
            "content": json.dumps(
                {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}
            ),
        },
    ]
    follow_up = litellm.completion(
        model="gpt-4o-mini",
        messages=follow_up_messages,
        tools=tools,
        stream=True,
    )
    for chunk in follow_up:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and getattr(delta, "content", None):
            print(delta.content, end="", flush=True)
    print()
    tracer_provider.force_flush()
