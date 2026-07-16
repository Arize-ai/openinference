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
        "service.name": "litellm-anthropic-reasoning-example",
        "openinference.project.name": "openinference-litellm-demo",
    }
)
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

MODEL = "anthropic/claude-sonnet-4-6"
THINKING = {"type": "enabled", "budget_tokens": 1024}

WEATHER_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city and country, e.g. Paris, France",
                }
            },
            "required": ["city"],
        },
    },
}


def get_weather(city: str) -> str:
    return json.dumps({"city": city, "forecast": "sunny", "temperature_c": 22})


def print_content(response: Any) -> None:
    message = response.choices[0].message
    print(getattr(message, "content", None))


def reasoning_completion() -> None:
    response = litellm.completion(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "What is 27 * 453? Think it through step by step.",
            }
        ],
        max_tokens=2048,
        thinking=THINKING,
    )
    print_content(response)


def reasoning_completion_stream() -> None:
    stream = litellm.completion(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "What is 27 * 453? Think it through step by step.",
            }
        ],
        max_tokens=2048,
        thinking=THINKING,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and getattr(delta, "content", None):
            print(delta.content, end="", flush=True)
    print()


def reasoning_completion_with_tool_call() -> None:
    messages: List[Any] = [
        {
            "role": "user",
            "content": "What is the weather in Paris? Think before using the tool.",
        }
    ]
    response = litellm.completion(
        model=MODEL,
        messages=messages,
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        max_tokens=2048,
        thinking=THINKING,
    )
    print(response)

    response_message = response.choices[0].message
    tool_calls = getattr(response_message, "tool_calls", None)
    if not tool_calls:
        return

    messages.append(response_message)
    for tool_call in tool_calls:
        function = tool_call.function
        if function.name != "get_weather":
            continue
        args = json.loads(function.arguments or "{}")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function.name,
                "content": get_weather(args["city"]),
            }
        )

    follow_up = litellm.completion(
        model=MODEL,
        messages=messages,
        tools=[WEATHER_TOOL],
        max_tokens=2048,
        thinking=THINKING,
    )
    print_content(follow_up)


async def async_reasoning_completion() -> None:
    response = await litellm.acompletion(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "What is 27 * 453? Think it through step by step.",
            }
        ],
        max_tokens=2048,
        thinking=THINKING,
    )
    print_content(response)


async def async_reasoning_completion_stream() -> None:
    stream = await litellm.acompletion(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "What is 27 * 453? Think it through step by step.",
            }
        ],
        max_tokens=2048,
        thinking=THINKING,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and getattr(delta, "content", None):
            print(delta.content, end="", flush=True)
    print()


async def async_reasoning_completion_with_tool_call() -> None:
    messages: List[Any] = [
        {
            "role": "user",
            "content": "What is the weather in Tokyo? Think before using the tool.",
        }
    ]
    response = await litellm.acompletion(
        model=MODEL,
        messages=messages,
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        max_tokens=2048,
        thinking=THINKING,
    )
    print(response)

    response_message = response.choices[0].message
    tool_calls = getattr(response_message, "tool_calls", None)
    if not tool_calls:
        return

    messages.append(response_message)
    for tool_call in tool_calls:
        function = tool_call.function
        if function.name != "get_weather":
            continue
        args = json.loads(function.arguments or "{}")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function.name,
                "content": get_weather(args["city"]),
            }
        )

    follow_up = await litellm.acompletion(
        model=MODEL,
        messages=messages,
        tools=[WEATHER_TOOL],
        max_tokens=2048,
        thinking=THINKING,
    )
    print_content(follow_up)


async def main() -> None:
    reasoning_completion()
    reasoning_completion_stream()
    reasoning_completion_with_tool_call()
    await async_reasoning_completion()
    await async_reasoning_completion_stream()
    await async_reasoning_completion_with_tool_call()
    tracer_provider.force_flush()


if __name__ == "__main__":
    asyncio.run(main())
