import os
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.types import Content, GenerateContentConfig, Part
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)


def run() -> None:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = GenerateContentConfig(
        system_instruction=(
            "You are a helpful assistant that can answer questions and help with tasks."
        )
    )
    file_path = "img.png"
    path = Path(file_path)
    image_bytes = path.read_bytes()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    content = Content(
        role="user",
        parts=[
            Part.from_text(text="Describe Image and write a poem about it."),
            image_part,
        ],
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=content,
        config=config,
    )
    print(response.text)


if __name__ == "__main__":
    run()
