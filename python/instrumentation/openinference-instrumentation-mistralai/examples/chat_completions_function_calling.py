from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage, FunctionCall, ToolCall, ToolChoice
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.mistralai import MistralAIInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace_api.set_tracer_provider(tracer_provider)

MistralAIInstrumentor().instrument()


if __name__ == "__main__":
    client = MistralClient()
    response = client.chat(
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
    message = response.choices[0].message
    print(message.tool_calls)

    response = client.chat(
        model="mistral-large-latest",
        messages=[
            ChatMessage(
                content="What's the weather like in San Francisco?",
                role="user",
            ),
            ChatMessage(
                content="",
                role="assistant",
                tool_calls=[
                    ToolCall(
                        function=FunctionCall(
                            name="get_weather", arguments='{"city": "San Francisco"}'
                        )
                    )
                ],
            ),
            ChatMessage(role="tool", name="get_weather", content='{"weather_category": "sunny"}'),
        ],
    )
    print(response.choices[0].message.content)
