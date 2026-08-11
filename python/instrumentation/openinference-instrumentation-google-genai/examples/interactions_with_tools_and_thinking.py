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

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_weather(location: str) -> str:
    """Gets the weather for a given location."""
    return f"The weather in {location} is sunny and 22°C."


weather_tool = {
    "type": "function",
    "name": "get_weather",
    "description": "Gets the current weather for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and country, e.g. Paris, France",
            }
        },
        "required": ["location"],
    },
}


def run() -> None:
    # thinking_level enables reasoning; the model may attach a reasoning_signature
    # to function_call steps, which the instrumentation captures as
    # tool_call.reasoning_signature on the span.
    interaction = client.interactions.create(
        model=MODEL,
        input="What is the weather in Paris and New Delhi?",
        tools=[weather_tool],
        generation_config={
            "thinking_level": "low",
            "thinking_summaries": "auto",
        },
    )
    print(interaction)


if __name__ == "__main__":
    run()
