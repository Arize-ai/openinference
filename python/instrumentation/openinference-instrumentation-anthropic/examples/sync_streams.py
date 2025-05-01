from anthropic import Anthropic
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.anthropic import AnthropicInstrumentor

# Configure HaystackInstrumentor with Phoenix endpoint
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

client = Anthropic()

with client.messages.stream(
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
    model="claude-3-7-sonnet-20250219",
) as stream:
    for text in stream:
        print(text, end="", flush=True)
