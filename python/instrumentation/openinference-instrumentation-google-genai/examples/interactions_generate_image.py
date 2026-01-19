
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


def run():
    client = genai.Client()

    interaction = client.interactions.create(
        model="gemini-2.5-flash-image",
        input="Create a simple flat illustration of a red apple on a white background. No text.",
        stream=True
    )
    for chunk in interaction:
        # print(chunk)
        pass


if __name__ == "__main__":
    run()