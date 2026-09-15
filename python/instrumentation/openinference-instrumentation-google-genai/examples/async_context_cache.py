import asyncio
import pathlib

from google import genai
from google.genai import types
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,  # type: ignore[import-not-found]
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)

client = genai.Client().aio

model = "gemini-2.5-flash"


async def create_text_cache():
    with open(pathlib.Path(__file__).parent / "story.txt", "r") as f:
        story_content = f.read()
        content = types.Content(
            parts=[types.Part(text=story_content)],
            role="user",
        )
        cache = await client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                display_name="ai developer context",
                system_instruction=(
                    "You are an expert Agent, and your job is to answer "
                    "the user query based on the Context you have access to."
                ),
                contents=[content],
                ttl="300s",
            ),
        )
        print(cache)
        return cache


if __name__ == "__main__":
    asyncio.run(create_text_cache())
