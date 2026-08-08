"""Example: tracing Anthropic's beta tool runner with OpenInference.

Runs Anthropic's ``client.beta.messages.tool_runner`` against a local tool
function so that tool use, tool calls, and the final answer flow through
the instrumented session. Spans are exported to a local Phoenix or any
OTLP-compatible collector on port 6006.

Run:

    ANTHROPIC_API_KEY=... python tool_runner.py
"""

import anthropic
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.anthropic import AnthropicInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)


def get_weather(location: str) -> str:
    """A local tool the runner invokes when the model asks for weather."""
    return f"The weather in {location} is sunny and 72 degrees."


client = anthropic.Anthropic(api_key="sk-proj-...")
runner = client.beta.messages.tool_runner(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    max_iterations=3,
    messages=[{"role": "user", "content": "What is the weather in Paris?"}],
    tools=[get_weather],
)

for message in runner:
    for block in message.content:
        if block.type == "text":
            print(block.text)
