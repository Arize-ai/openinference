from anthropic import Anthropic
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.anthropic import AnthropicInstrumentor

# Configure AnthropicInstrumentor with Phoenix endpoint
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

# The legacy Text Completions API (client.completions) was removed in anthropic v1;
# use the Messages API instead.
client = Anthropic()

resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": "how does a court case get to the Supreme Court?",
        }
    ],
)

print(resp)
