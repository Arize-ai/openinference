import asyncio

import openai
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.openai import OpenAIInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


async def responses_create(**kwargs):
    client = openai.AsyncOpenAI()
    response = await client.responses.create(**kwargs)
    async for event in response:
        if event.type == "response.output_text.delta":
            print(event.delta, end="")


asyncio.run(
    responses_create(
        model="gpt-4o-mini",
        instructions="You are an agent to generate stories.",
        input=[
            {
                "role": "user",
                "content": "Tell me a three sentence bedtime story about a unicorn.",
            }
        ],
        stream=True,
    )
)
