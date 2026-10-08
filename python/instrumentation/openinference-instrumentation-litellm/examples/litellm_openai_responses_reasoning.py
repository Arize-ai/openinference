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
        "service.name": "litellm-openai-responses-reasoning-example",
        "openinference.project.name": "openinference-litellm-demo",
    }
)
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

MODEL = "openai/o4-mini"

WEATHER_TOOL: Dict[str, Any] = {
    "type": "function",
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
        "additionalProperties": False,
    },
    "strict": True,
}


def get_weather(city: str) -> str:
    return json.dumps({"city": city, "forecast": "sunny", "temperature_c": 22})


def print_response_text(response: Any) -> None:
    if output_text := getattr(response, "output_text", None):
        print(output_text)
    else:
        print(response)


def reasoning_response() -> None:
    response = litellm.responses(
        model=MODEL,
        input="Write a one-sentence bedtime story about a lighthouse.",
        reasoning={"effort": "low", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )
    print_response_text(response)


def reasoning_response_stream() -> None:
    stream = litellm.responses(
        model=MODEL,
        input="Write a one-sentence bedtime story about a lighthouse.",
        reasoning={"effort": "low", "summary": "auto"},
        include=["reasoning.encrypted_content"],
        stream=True,
    )
    for event in stream:
        print(event)


def reasoning_response_with_tool_call() -> None:
    input_messages: List[Any] = [
        {"role": "user", "content": "What is the weather in Paris? Think before using tools."}
    ]
    response = litellm.responses(
        model=MODEL,
        input=input_messages,
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        reasoning={"effort": "low", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )
    print(response)

    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "reasoning":
            input_messages.append(item)
            continue
        if getattr(item, "type", None) != "function_call":
            continue
        if getattr(item, "name", None) != "get_weather":
            continue

        args = json.loads(getattr(item, "arguments", "{}"))
        input_messages.append(item)
        input_messages.append(
            {
                "type": "function_call_output",
                "call_id": getattr(item, "call_id"),
                "output": get_weather(args["city"]),
            }
        )

    if len(input_messages) == 1:
        return

    follow_up = litellm.responses(
        model=MODEL,
        input=input_messages,
        tools=[WEATHER_TOOL],
        reasoning={"effort": "low", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )
    print_response_text(follow_up)


async def async_reasoning_response() -> None:
    response = await litellm.aresponses(
        model=MODEL,
        input="Write a one-sentence bedtime story about a lighthouse.",
        reasoning={"effort": "low", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )
    print_response_text(response)


async def async_reasoning_response_stream() -> None:
    stream = await litellm.aresponses(
        model=MODEL,
        input="Write a one-sentence bedtime story about a lighthouse.",
        reasoning={"effort": "low", "summary": "auto"},
        include=["reasoning.encrypted_content"],
        stream=True,
    )
    async for event in stream:
        print(event)


async def async_reasoning_response_with_tool_call() -> None:
    input_messages: List[Any] = [
        {"role": "user", "content": "What is the weather in Tokyo? Think before using tools."}
    ]
    response = await litellm.aresponses(
        model=MODEL,
        input=input_messages,
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        reasoning={"effort": "low", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )
    print(response)

    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "reasoning":
            input_messages.append(item)
            continue
        if getattr(item, "type", None) != "function_call":
            continue
        if getattr(item, "name", None) != "get_weather":
            continue
        args = json.loads(getattr(item, "arguments", "{}"))
        input_messages.append(item)
        input_messages.append(
            {
                "type": "function_call_output",
                "call_id": getattr(item, "call_id"),
                "output": get_weather(args["city"]),
            }
        )

    if len(input_messages) == 1:
        return

    follow_up = await litellm.aresponses(
        model=MODEL,
        input=input_messages,
        tools=[WEATHER_TOOL],
        reasoning={"effort": "low", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )
    print_response_text(follow_up)


async def main() -> None:
    reasoning_response()
    reasoning_response_stream()
    reasoning_response_with_tool_call()
    await async_reasoning_response()
    await async_reasoning_response_stream()
    await async_reasoning_response_with_tool_call()
    tracer_provider.force_flush()


if __name__ == "__main__":
    asyncio.run(main())
