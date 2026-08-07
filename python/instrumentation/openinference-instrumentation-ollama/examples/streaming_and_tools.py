import os

import ollama
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation import using_attributes
from openinference.instrumentation.ollama import OllamaInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

OllamaInstrumentor().instrument(tracer_provider=tracer_provider)

MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


if __name__ == "__main__":
    # Streaming chat: the span finishes when the stream is exhausted and
    # records the accumulated output and token counts.
    with using_attributes(session_id="example-session", user_id="example-user"):
        for chunk in ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            stream=True,
        ):
            print(chunk.message.content, end="", flush=True)
    print()

    # Tool calling with a plain Python function: the tool's JSON schema is
    # recorded on the span, along with any tool calls in the response.
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 2 + 3? Use the tool."}],
        tools=[add_two_numbers],
    )
    for tool_call in response.message.tool_calls or []:
        print(f"{tool_call.function.name}({dict(tool_call.function.arguments)})")
