import json

from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from openinference.instrumentation.haystack import HaystackInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

HaystackInstrumentor().instrument(tracer_provider=tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": (
                            "The temperature unit to use. Infer this from the users location."
                        ),
                    },
                },
                "required": ["location", "format"],
            },
        },
    }
]
messages = [ChatMessage.from_user("What's the weather like in Berlin?")]
llm = OpenAIChatGenerator(model="gpt-4o")
pipe = Pipeline()
pipe.add_component("llm", llm)
response = pipe.run({"llm": {"messages": messages, "generation_kwargs": {"tools": tools}}})
response_msg = response["llm"]["replies"][0]
messages.append(response_msg)
weather_response = [
    {
        "id": "response_uhGNifLfopt5JrCUxXw1L3zo",
        "status": "success",
        "function": {
            "name": "get_current_weather",
            "arguments": {"location": "Berlin", "format": "celsius"},
        },
        "data": {
            "location": "Berlin",
            "temperature": 18,
            "weather_condition": "Partly Cloudy",
            "humidity": "60%",
            "wind_speed": "15 km/h",
            "observation_time": "2024-03-05T14:00:00Z",
        },
    }
]
messages.append(
    ChatMessage.from_function(content=json.dumps(weather_response), name="get_current_weather")
)
response = pipe.run({"llm": {"messages": messages, "generation_kwargs": {"tools": tools}}})
print(f"{response["llm"]["replies"][0]=}")
