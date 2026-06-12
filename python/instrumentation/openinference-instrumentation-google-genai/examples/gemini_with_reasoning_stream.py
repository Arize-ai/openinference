import os

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

# Make sure to set the GEMINI_API_KEY environment variable

MODEL = "gemini-2.5-flash"


def generate_with_reasoning_stream() -> None:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    stream = client.models.generate_content_stream(
        model=MODEL,
        contents="Tell bed storey for 5 years old boy.",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=512,
                # thinking_level=ThinkingLevel.LOW,
            ),
        ),
    )

    for chunk in stream:
        print(chunk)


if __name__ == "__main__":
    generate_with_reasoning_stream()
