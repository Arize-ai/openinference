import os
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.types import Content, GenerateContentConfig, Part
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


def send_message_multi_turn() -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = GenerateContentConfig(system_instruction="You are a helpful assistant.")
    history = [
        Content(
            parts=[
                Part.from_text(text="What is the capital of France?"),
                Part.from_text(text="What is the capital of India?"),
            ],
            role="user",
        ),
        Content(
            parts=[
                Part.from_text(
                    text="The capital of France is Paris. The capital of India is New Delhi."
                )
            ],
            role="model",
        ),
        Content(parts=[Part.from_text(text="What is Sum of 1, 1")], role="user"),
        Content(parts=[Part.from_text(text="Answer is 2")], role="model"),
    ]
    chat = client.chats.create(model="gemini-2.5-flash", config=config, history=history)
    file_path = "img.png"
    path = Path(file_path)
    image_bytes = path.read_bytes()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    text_part = Part.from_text(text="Describe Image and write a poem about it.")
    response = chat.send_message(
        [
            text_part,
            image_part,
        ]
    )
    return response.text or ""


if __name__ == "__main__":
    GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    response1 = send_message_multi_turn()
