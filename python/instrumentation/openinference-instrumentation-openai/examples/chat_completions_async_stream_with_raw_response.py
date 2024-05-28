import asyncio

import openai
from openai.types.chat import ChatCompletionStreamOptionsParam
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace_api.set_tracer_provider(tracer_provider)

OpenAIInstrumentor().instrument()


async def chat_completions(**kwargs):
    client = openai.AsyncOpenAI()
    response = await client.chat.completions.with_raw_response.create(**kwargs)
    async for chunk in response.parse():
        if chunk.choices and (content := chunk.choices[0].delta.content):
            print(content, end="")


if __name__ == "__main__":
    asyncio.run(
        chat_completions(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Write a haiku."}],
            max_tokens=20,
            stream=True,
            stream_options=ChatCompletionStreamOptionsParam(include_usage=True),
        ),
    )
