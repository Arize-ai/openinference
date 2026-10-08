import json

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from together import Together

from openinference.instrumentation.together import TogetherInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

TogetherInstrumentor().instrument(tracer_provider=tracer_provider)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city, e.g. Paris"},
                },
                "required": ["city"],
            },
        },
    }
]


if __name__ == "__main__":
    client = Together()
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": "What is the weather in Paris right now?"}],
        tools=tools,
    )
    message = response.choices[0].message
    if message.tool_calls:
        for tool_call in message.tool_calls:
            print(f"tool call: {tool_call.function.name}({tool_call.function.arguments})")
            print(json.loads(tool_call.function.arguments))
    else:
        print(message.content)
