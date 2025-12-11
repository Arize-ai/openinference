"""
Basic example of using Strands instrumentation.

This example demonstrates how to instrument a simple Strands agent
using OpenAI and export traces to an OpenTelemetry collector.

Before running:
    export OPENAI_API_KEY='your-api-key-here'
"""

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from strands import Agent, tool
from strands.models.openai import OpenAIModel

from openinference.instrumentation import using_attributes
from openinference.instrumentation.strands import StrandsInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

StrandsInstrumentor().instrument(tracer_provider=tracer_provider)


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: The name of the city to get weather for

    Returns:
        Weather information for the city
    """
    # Simulate weather lookup
    weather_data = {
        "San Francisco": "sunny, 72°F",
        "New York": "cloudy, 65°F",
        "London": "rainy, 58°F",
        "Tokyo": "clear, 68°F",
    }

    weather = weather_data.get(city, "unknown")

    return {
        "status": "success",
        "content": [{"text": f"The weather in {city} is {weather}"}],
    }


if __name__ == "__main__":
    model = OpenAIModel(model_id="gpt-4o-mini")
    agent = Agent(
        name="Weather Assistant",
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful weather assistant.",
    )

    with using_attributes(
        session_id="my-test-session",
        user_id="my-test-user",
        metadata={
            "example": "basic_agent",
            "framework": "strands",
        },
        tags=["weather", "demo"],
    ):
        result = agent("What's the weather in San Francisco?")
        print(result.message)
