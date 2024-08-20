import openai
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.openai import OpenAIInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


if __name__ == "__main__":
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
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
            },
        ],
    )
    message = response.choices[0].message
    assert (tool_calls := message.tool_calls)
    tool_call_id = tool_calls[0].id
    client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"content": "What's the weather like in San Francisco?", "role": "user"},
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="",
                tool_calls=[
                    ChatCompletionMessageToolCallParam(
                        id=tool_call_id,
                        function={"name": "get_weather", "arguments": '{"city": "San Francisco"}'},
                        type="function",
                    ),
                ],
            ),
            ChatCompletionToolMessageParam(
                content='{"weather_category": "sunny"}', role="tool", tool_call_id=tool_call_id
            ),
        ],
    )
