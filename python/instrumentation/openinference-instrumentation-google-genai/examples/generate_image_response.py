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

    prompt = "Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme"
    response = client.models.generate_content(
        model="gemini-3-pro-image",
        contents=[prompt],
    )

    for part in response.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = part.as_image()
            image.save("generated_image.png")


if __name__ == "__main__":
    GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    generate_image()
