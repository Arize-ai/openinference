import asyncio
import gc

from anthropic import Anthropic, AsyncAnthropic
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation import using_metadata
from openinference.instrumentation.anthropic import AnthropicInstrumentor

# Configure AnthropicInstrumentor with Phoenix endpoint
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

client = Anthropic()
async_client = AsyncAnthropic()

request = dict(
    max_tokens=200,
    messages=[{"role": "user", "content": "Count from 1 to 30, one number per line."}],
    model="claude-sonnet-4-6",
    stream=True,
)


def print_text(event):
    if event.type == "content_block_delta" and event.delta.type == "text_delta":
        print(event.delta.text, end="", flush=True)


# Each stream below is left before the response is complete. Its span still shows up in
# Phoenix with the text received so far.


def close_early():
    stream = client.messages.create(**request)
    for i, event in enumerate(stream):
        print_text(event)
        if i == 5:
            break
    stream.close()
    print()


def drop_early():
    stream = client.messages.create(**request)
    for i, event in enumerate(stream):
        print_text(event)
        if i == 5:
            break
    del stream
    gc.collect()
    print()


async def async_close_early():
    stream = await async_client.messages.create(**request)
    i = 0
    async for event in stream:
        print_text(event)
        i += 1
        if i == 6:
            break
    await stream.close()
    print()


if __name__ == "__main__":
    with using_metadata({"scenario": "close_early"}):
        close_early()
    with using_metadata({"scenario": "drop_early"}):
        drop_early()
    with using_metadata({"scenario": "async_close_early"}):
        asyncio.run(async_close_early())
