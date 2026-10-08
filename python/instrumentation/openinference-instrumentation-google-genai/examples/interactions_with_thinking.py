import os

from google import genai
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


def interactions_with_thinking() -> None:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    interaction = client.interactions.create(
        model=MODEL,
        input="Explain why the sky is blue in two sentences.",
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 1024,
            "thinking_level": "low",
            "thinking_summaries": "auto",
        },
        stream=True,
    )

    print("=== Response ===")
    for chunk in interaction:
        print(chunk)
    print()


if __name__ == "__main__":
    interactions_with_thinking()
