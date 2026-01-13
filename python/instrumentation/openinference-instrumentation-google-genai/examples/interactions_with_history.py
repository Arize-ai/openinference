
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


if __name__ == "__main__":
    client = genai.Client()
    conversation_history = [
        {
            "role": "user",
            "content": "What are the three largest cities in Spain?"
        }
    ]

    interaction1 = client.interactions.create(
        model="gemini-3-flash-preview",
        input=conversation_history
    )
    print(f"Model: {interaction1.outputs[-1].text}")
    conversation_history.append({"role": "model", "content": interaction1.outputs})
    conversation_history.append({
        "role": "user",
        "content": "What is the most famous landmark in the second one?"
    })

    interaction2 = client.interactions.create(
        model="gemini-3-flash-preview",
        input=conversation_history
    )

    print(f"Model: {interaction2.outputs[-1].text}")