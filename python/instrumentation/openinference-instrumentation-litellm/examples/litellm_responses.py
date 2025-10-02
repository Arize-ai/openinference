import asyncio

import litellm
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.litellm import LiteLLMInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)


stream = True


def responses_image_input():
    response = litellm.responses(
        model="openai/gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is in this image?"},
                    {
                        "type": "input_image",
                        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                ],
            }
        ],
        stream=stream,
    )
    print(list(response))


def responses_file_input():
    response = litellm.responses(
        model="openai/gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is in this file?"},
                    {
                        "type": "input_file",
                        "file_url": "https://www.berkshirehathaway.com/letters/2024ltr.pdf",
                    },
                ],
            }
        ],
        stream=stream,
    )
    print(list(response))


def websearch_input():
    response = litellm.responses(
        model="openai/gpt-4.1",
        tools=[{"type": "web_search_preview"}],
        input="What was a positive news story from today?",
        stream=stream,
    )
    print(list(response))


def function_call():
    tools = [
        {
            "type": "function",
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
                "required": ["location", "unit"],
            },
        }
    ]
    response = litellm.responses(
        model="openai/gpt-4.1",
        tools=tools,
        input="What is the weather like in Boston today?",
        tool_choice="auto",
        stream=stream,
    )
    print(list(response))


def reasoning_input():
    response = litellm.responses(
        model="openai/o3-mini",
        input="How much wood would a woodchuck chuck?",
        reasoning={"effort": "high"},
        stream=stream,
    )
    print(list(response))


async def async_responses_image_input():
    response = await litellm.aresponses(
        model="openai/gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is in this image?"},
                    {
                        "type": "input_image",
                        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                ],
            }
        ],
        stream=stream,
    )
    print(list(response))


async def async_responses_file_input():
    response = await litellm.aresponses(
        model="openai/gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is in this file?"},
                    {
                        "type": "input_file",
                        "file_url": "https://www.berkshirehathaway.com/letters/2024ltr.pdf",
                    },
                ],
            }
        ],
        stream=stream,
    )
    print(list(response))


async def async_websearch_input():
    response = await litellm.aresponses(
        model="openai/gpt-4.1",
        tools=[{"type": "web_search_preview"}],
        input="What was a positive news story from today?",
        stream=stream,
    )
    print(list(response))


async def async_function_call():
    tools = [
        {
            "type": "function",
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
                "required": ["location", "unit"],
            },
        }
    ]
    response = await litellm.aresponses(
        model="openai/gpt-4.1",
        tools=tools,
        input="What is the weather like in Boston today?",
        tool_choice="auto",
        stream=stream,
    )
    print(list(response))


async def async_reasoning_input():
    response = await litellm.aresponses(
        model="openai/o3-mini",
        input="How much wood would a woodchuck chuck?",
        reasoning={"effort": "high"},
        stream=stream,
    )
    print(list(response))


if __name__ == "__main__":
    responses_image_input()
    responses_file_input()
    websearch_input()
    function_call()
    reasoning_input()
    asyncio.run(async_responses_image_input())
    asyncio.run(async_responses_file_input())
    asyncio.run(async_websearch_input())
    asyncio.run(async_function_call())
    asyncio.run(async_reasoning_input())
