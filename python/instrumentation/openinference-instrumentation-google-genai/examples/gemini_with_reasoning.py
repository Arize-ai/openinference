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

MODEL = "gemini-2.5-flash-preview-05-20"


def generate_with_reasoning() -> None:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model=MODEL,
        contents="What is the capital of France? Show your reasoning.",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        ),
    )
    print(response.text)

    # Collect all parts from the model response (thought + text)
    prior_parts: list[types.Part] = []
    if response.candidates and response.candidates[0].content:
        prior_parts = list(response.candidates[0].content.parts or [])

    if prior_parts:
        follow_up = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text="What is the capital of France? Show your reasoning."
                        )
                    ],
                ),
                types.Content(role="model", parts=prior_parts),
                types.Content(
                    role="user",
                    parts=[types.Part(text="And what country is Paris in?")],
                ),
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=512),
            ),
        )
        print(follow_up.text)


if __name__ == "__main__":
    generate_with_reasoning()
