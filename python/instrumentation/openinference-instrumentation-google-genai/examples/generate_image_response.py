from google import genai
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,  # type: ignore[import-not-found]
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

endpoint = "http://0.0.0.0:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))


def generate_image() -> None:
    client = genai.Client()
    response = client.models.generate_content_stream(
        model="gemini-2.5-flash-image",
        contents="Create a simple flat illustration of a red apple on a white background. No text."
        " Minimal detail. Low complexity.",
    )
    for res in response:
        print(res)


if __name__ == "__main__":
    GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    generate_image()
