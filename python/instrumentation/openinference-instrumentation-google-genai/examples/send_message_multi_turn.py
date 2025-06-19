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


def send_message_multi_turn() -> tuple[str, str]:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    chat = client.chats.create(model="gemini-2.0-flash-001")
    response1 = chat.send_message("What is the capital of France?")
    response2 = chat.send_message("Why is the sky blue?")

    return response1.text or "", response2.text or ""


if __name__ == "__main__":
    GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    response1, response2 = send_message_multi_turn()
    print(response1)
    print(response2)
