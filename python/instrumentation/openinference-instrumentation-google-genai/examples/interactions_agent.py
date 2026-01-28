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


def run():
    client = genai.Client()

    interaction = client.interactions.create(
        input="Research the history of the Google TPUs with a focus on 2025 and 2026.",
        agent="deep-research-pro-preview-12-2025",
        background=True,
    )
    print(interaction)


if __name__ == "__main__":
    run()
