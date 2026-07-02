import asyncio

import litellm
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.litellm import LiteLLMInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
litellm.drop_params = True

tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)


async def run():
    response = litellm.completion(
        model="openai/responses/gpt-5-mini",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        reasoning_effort={"effort": "low", "summary": "detailed"},  # Explicit control
    )
    print("response\n", response)

    response = litellm.completion(
        model="anthropic/claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 1024},
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=8096,
    )

    print("Anthropic response\n", response)


if __name__ == "__main__":
    asyncio.run(run())
