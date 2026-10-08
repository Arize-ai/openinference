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
    response_stream = client.chat.stream(
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

    tool_call_arguments = ""
    for event in response_stream:
        if event.data.choices:
            delta = event.data.choices[0].delta

            if delta.tool_calls:
                tool_call = delta.tool_calls[0]
                print(tool_call)

                if tool_call.function is not None and tool_call.function.arguments is not None:
                    tool_call_arguments += tool_call.function.arguments

    print("Collected arguments:", tool_call_arguments)
