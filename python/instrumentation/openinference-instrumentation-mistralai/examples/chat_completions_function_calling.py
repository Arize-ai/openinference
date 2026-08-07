import os

from mistralai.client import Mistral
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.mistralai import MistralAIInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

MistralAIInstrumentor().instrument(tracer_provider=tracer_provider)


if __name__ == "__main__":
    client = Mistral(
        api_key=os.getenv("MISTRAL_API_KEY", ""),
    )
    response = client.chat.complete(
        model="mistral-large-latest",
        tool_choice="any",
        tools=[
            {
                "type": "function",
                "function": {
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
            },
        ],
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in San Francisco?",
            }
        ],
    )

    message = response.choices[0].message
    print(message.tool_calls)
    tool_call = message.tool_calls[0]

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in San Francisco?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "get_weather",
                "content": '{"weather_category":"sunny"}',
            },
        ],
    )
    print(response.choices[0].message.content)
