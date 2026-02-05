"""Basic example of using Microsoft Agent Framework with OpenInference instrumentation.

This example demonstrates how to use the AgentFrameworkToOpenInferenceProcessor
to transform Microsoft Agent Framework's native OpenTelemetry spans into OpenInference format.
"""

import asyncio
import os

from openinference.instrumentation.agent_framework import (
    AgentFrameworkToOpenInferenceProcessor,
)


async def main():
    """Run the basic agent example."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please run: export OPENAI_API_KEY='your-api-key-here'")
        return

    print("=" * 60)
    print("Basic Microsoft Agent Framework with OpenInference Instrumentation")
    print("=" * 60)
    print()

    # Setup telemetry using Microsoft Agent Framework's observability
    print("Setting up telemetry...")

    from agent_framework.observability import configure_otel_providers
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    # Configure MS Agent Framework's native telemetry with OTLP exporter
    # Set OTEL_EXPORTER_OTLP_ENDPOINT env var or it will use default
    # For Phoenix: export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006/v1/traces
    configure_otel_providers(enable_sensitive_data=True)

    # Add OpenInference processor to transform spans
    tracer_provider = trace.get_tracer_provider()
    if isinstance(tracer_provider, TracerProvider):
        tracer_provider.add_span_processor(AgentFrameworkToOpenInferenceProcessor(debug=False))

    print("Telemetry configured")
    print()

    # Create agent
    print("Creating agent...")
    from agent_framework.openai import OpenAIChatClient

    client = OpenAIChatClient(model_id="gpt-4o-mini")
    agent = client.create_agent(
        name="WeatherAssistant",
        instructions="You are a helpful assistant. Be concise in your responses.",
    )
    print("Agent created")
    print()

    # Run agent
    print("Running agent query: 'What is the capital of France? Answer in one sentence.'")
    print()

    result = await agent.run("What is the capital of France? Answer in one sentence.")

    print("=" * 60)
    print("Result:")
    print("=" * 60)
    print(result.text)
    print()
    print(f"Token Usage: {result.usage_details}")
    print()
    print("Done! Check Phoenix at http://localhost:6006 to see traces")


if __name__ == "__main__":
    asyncio.run(main())
