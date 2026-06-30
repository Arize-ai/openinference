import json
from typing import Any

import openai
import requests
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

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

input_messages = [{"role": "user", "content": "What's the weather like in Paris & Delhi today?"}]

response = client.responses.create(model="gpt-4o-mini", input=input_messages, tools=tools)
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
    model="gpt-4o-mini",
    input=input_messages,
    tools=tools,
)
print(response_2.output_text)
