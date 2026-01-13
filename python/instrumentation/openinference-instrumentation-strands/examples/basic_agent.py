"""Basic example of using Strands with OpenInference instrumentation.

This example demonstrates how to use the StrandsToOpenInferenceProcessor
to transform Strands' native OpenTelemetry spans into OpenInference format.
"""

import os

from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands.telemetry import StrandsTelemetry

from openinference.instrumentation.strands import StrandsToOpenInferenceProcessor


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: The name of the city
    """
    return {
        "status": "success",
        "content": [{"text": f"The weather in {city} is sunny and 72°F."}],
    }


def main():
    """Run the basic agent example."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please run: export OPENAI_API_KEY='your-api-key-here'")
        return

    print("=" * 60)
    print("Basic Strands Agent with OpenInference Instrumentation")
    print("=" * 60)
    print()

    # Setup Strands' native telemetry
    print("📡 Setting up telemetry...")
    telemetry = StrandsTelemetry()

    # Export to Phoenix (or other OTLP endpoint)
    telemetry.setup_otlp_exporter(endpoint="http://127.0.0.1:6006/v1/traces")

    # Optional: Also log to console for debugging
    # telemetry.setup_console_exporter()

    # Add OpenInference processor to transform spans
    telemetry.tracer_provider.add_span_processor(
        StrandsToOpenInferenceProcessor(debug=False)  # Set debug=True for verbose output
    )

    print("✅ Telemetry configured")
    print()

    # Create agent
    print("🤖 Creating agent...")
    model = OpenAIModel(model_id="gpt-4o-mini")
    agent = Agent(
        name="WeatherAssistant",
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful weather assistant.",
    )
    print("✅ Agent created")
    print()

    # Run agent
    print("💬 Running agent query: 'What's the weather in San Francisco?'")
    print()

    result = agent("What's the weather in San Francisco?")

    print("=" * 60)
    print("Result:")
    print("=" * 60)
    print(result)
    print()
    print("✨ Done! Check Phoenix at http://127.0.0.1:6006 to see traces")


if __name__ == "__main__":
    main()
