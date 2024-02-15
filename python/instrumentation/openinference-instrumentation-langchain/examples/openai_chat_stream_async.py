import asyncio

from langchain_openai import ChatOpenAI
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
trace_api.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

LangChainInstrumentor().instrument()


async def main():
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=20)
    async for chunk in chat.astream("Write a haiku."):
        print(chunk.content, end="", flush=True)
        raise asyncio.CancelledError("Cancelled")


if __name__ == "__main__":
    asyncio.run(main())
