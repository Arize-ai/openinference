import asyncio
import os

from google import genai
from google.genai.types import Content, GenerateContentConfig, Part
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

# Make sure to set the GEMINI_API_KEY environment variable


async def run():
    client = genai.Client().aio

    initial_interaction = await client.interactions.create(
        input="Research the history of the Google TPUs with a focus on 2025 and 2026.",

    )
    while True:
        interaction = await client.interactions.get(initial_interaction.id)
        print(f"Status: {interaction.status}")

        if interaction.status == "completed":
            print("\nFinal Report:\n", interaction.outputs[-1].text)
            break
        elif interaction.status in ["failed", "cancelled"]:
            print(f"Failed with status: {interaction.status}")
            break
        import time
        time.sleep(10)


if __name__ == "__main__":
    client = genai.Client()

    initial_interaction = client.interactions.create(
        input="Research the history of the Google TPUs with a focus on 2025 and 2026.",
        agent="deep-research-pro-preview-12-2025",
        background=True
    )
    while True:
        interaction = client.interactions.get(initial_interaction.id)
        print(f"Status: {interaction.status}")

        if interaction.status == "completed":
            print("\nFinal Report:\n", interaction.outputs[-1].text)
            break
        elif interaction.status in ["failed", "cancelled"]:
            print(f"Failed with status: {interaction.status}")
            break
        import time
        time.sleep(10)
