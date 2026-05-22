import json

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.agno import AgnoInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

AgnoInstrumentor().instrument(tracer_provider=tracer_provider)


def get_weather(city: str) -> str:
    """Return current weather for a city."""
    return f"Sunny, 22°C in {city}."

if __name__ == "__main__":
    for event in Agent(name="OpenAI Agent", model=OpenAIChat(id="gpt-4o-mini"), tools=[get_weather]).run(
        "What is the weather in Paris and New Delhi?", stream=True
    ):
        print(event)

    print("\n=== Anthropic ===")
    for event in Agent(name="Anthropic Agent", model=Claude(id="claude-haiku-4-5-20251001"), tools=[get_weather]).run(
        "What is the weather in Paris and New Delhi?", stream=True
    ):
        print(event)

    print("\n=== Gemini ===")
    for event in Agent(name="Gemini Agent", model=Gemini(id="gemini-2.0-flash"), tools=[get_weather]).run(
        "What is the weather in Paris and New Delhi?", stream=True
    ):
        print(event)
