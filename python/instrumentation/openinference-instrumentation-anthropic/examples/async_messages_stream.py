import asyncio

from anthropic import AsyncAnthropic
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.anthropic import AnthropicInstrumentor

# Configure AnthropicInstrumentor with Phoenix endpoint
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

client = AsyncAnthropic()


async def main():
    async with client.messages.stream(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}],
        model="claude-sonnet-4-6",
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
        print()

    stream = await client.messages.create(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}],
        model="claude-sonnet-4-6",
        stream=True,
    )
    async for event in stream:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            print(event.delta.text, end="", flush=True)


asyncio.run(main())
