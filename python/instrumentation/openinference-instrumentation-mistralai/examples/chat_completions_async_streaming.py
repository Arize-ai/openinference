import asyncio

from mistralai import Mistral
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.mistralai import MistralAIInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

MistralAIInstrumentor().instrument(tracer_provider=tracer_provider)


async def run_async_streaming_chat_completion() -> None:
    client = Mistral(
        api_key="redacted",
    )
    response_stream = await client.chat.stream_async(
        model="mistral-small-latest",
        messages=[
            {
                "content": "Who won the World Cup in 2018?",
                "role": "user",
            },
        ],
    )
    async for chunk in response_stream:
        print(chunk)
    print()


if __name__ == "__main__":
    asyncio.run(run_async_streaming_chat_completion())
