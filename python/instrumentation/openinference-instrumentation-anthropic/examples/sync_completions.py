import anthropic
from anthropic import Anthropic
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Configure HaystackInstrumentor with Phoenix endpoint
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

client = Anthropic()

prompt = (
    f"{anthropic.HUMAN_PROMPT}"
    f" how does a court case get to the Supreme Court?"
    f" {anthropic.AI_PROMPT}"
)

resp = client.completions.create(
    model="claude-2.1",
    prompt=prompt,
    max_tokens_to_sample=1000,
)

print(resp)
