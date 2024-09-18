from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage, ToolChoice
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
    client = MistralClient()
    response_stream = client.chat_stream(
        model="mistral-large-latest",
        tool_choice=ToolChoice.any,
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
            ChatMessage(
                content="What's the weather like in San Francisco?",
                role="user",
            )
        ],
    )
    tool_call_arguments = ""
    for chunk in response_stream:
        if (tool_calls := chunk.choices[0].delta.tool_calls) is not None:
            tool_call = tool_calls[0]
            print(tool_call)
