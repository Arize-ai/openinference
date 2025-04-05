import json
from typing import Any

import openai
import requests
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation import using_attributes
from openinference.instrumentation.openai import OpenAIInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


def get_weather(latitude: str, longitude: str) -> Any:
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]["temperature_2m"]


def tool_calling_implementation():
    client = openai.OpenAI()
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {"latitude": {"type": "number"}, "longitude": {"type": "number"}},
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]

    input_messages = [
        {"role": "user", "content": "What's the weather like in Paris & Delhi today?"}
    ]

    response = client.responses.create(model="gpt-4o", input=input_messages, tools=tools)
    for tool_call in response.output:
        if tool_call.type != "function_call":
            continue
        args = json.loads(tool_call.arguments)
        if tool_call.name != "get_weather":
            continue
        result = get_weather(args["latitude"], args["longitude"])
        input_messages.append(tool_call)  # append model's function call message
        input_messages.append(
            {  # append result message
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": str(result),
            }
        )

    response_2 = client.responses.create(
        model="gpt-4o",
        input=input_messages,
        tools=tools,
    )
    print(response_2.output_text)


if __name__ == "__main__":
    with using_attributes(
        session_id="my-test-session",
        user_id="my-test-user",
        metadata={
            "test-int": 1,
            "test-str": "string",
            "test-list": [1, 2, 3],
            "test-dict": {
                "key-1": "val-1",
                "key-2": "val-2",
            },
        },
        tags=["tag-1", "tag-2"],
        prompt_template="Who won the soccer match in {city} on {date}",
        prompt_template_version="v1.0",
        prompt_template_variables={
            "city": "Johannesburg",
            "date": "July 11th",
        },
    ):
        tool_calling_implementation()
