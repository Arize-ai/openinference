"""Stream from the beta Messages API with adaptive thinking.

Two things here are the current shape on anthropic 1.0, where the legacy Text
Completions API of sync_completions.py no longer exists:

- `client.beta.messages` is the beta Messages API. It produces a
  `beta.messages.stream` span, separate from the `messages.stream` span that
  messages_stream.py produces.
- `thinking={"type": "adaptive"}` lets the model decide how much to think.
  The older `{"type": "enabled", "budget_tokens": N}` form is deprecated, and
  current models reject it. Reasoning arrives as `reasoning` content blocks on
  the span, so pass `display` to get readable text rather than empty blocks.
"""

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

with client.beta.messages.stream(
    max_tokens=2048,
    messages=[
        {
            "role": "user",
            "content": "A train leaves at 06:12 at 47 km/h. A second leaves at 07:48 at"
            " 71 km/h on the same track. When does the second catch the first?",
        }
    ],
    model="claude-sonnet-4-6",
    thinking={"type": "adaptive", "display": "summarized"},
) as stream:
    for event in stream:
        if event.type == "content_block_delta":
            if event.delta.type == "thinking_delta":
                print(event.delta.thinking, end="", flush=True)
            elif event.delta.type == "text_delta":
                print(event.delta.text, end="", flush=True)
print()
