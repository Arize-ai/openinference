from pydantic import BaseModel, Field
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

client = Anthropic()


class SimpleResponse(BaseModel):
    """Simple structured output model for testing beta.messages.parse()."""

    answer: str = Field(description="A brief answer to the question")
    confidence: str = Field(description="Confidence level: high, medium, or low")


response = client.beta.messages.parse(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "What is 2 + 2? Provide a brief answer and your confidence level.",
        }
    ],
    model="claude-sonnet-4-5-20250929",
    output_format=SimpleResponse,
)
print(response.parsed_output)
