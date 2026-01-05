import os

from google import genai
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,  # type: ignore[import-not-found]
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

endpoint = "http://0.0.0.0:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))


def generate_image() -> None:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents="Create a picture of a futuristic banana with neon lights in a cyberpunk city.",
    )

    for part in response.parts:
        if part.inline_data:
            image = part.as_image()
            image.show()


if __name__ == "__main__":
    GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    generate_image()
