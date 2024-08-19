import asyncio

from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.mistralai import MistralAIInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace_api.set_tracer_provider(tracer_provider)

MistralAIInstrumentor().instrument()


async def run_async_streaming_chat_completion() -> None:
    client = MistralAsyncClient()
    response_stream = client.chat_stream(
        model="mistral-large-latest",
        messages=[
            ChatMessage(
                content="Who won the World Cup in 2018?",
                role="user",
            )
        ],
    )
    async for chunk in response_stream:
        print(chunk.choices[0].delta.content, end="")
    print()


if __name__ == "__main__":
    asyncio.run(run_async_streaming_chat_completion())
