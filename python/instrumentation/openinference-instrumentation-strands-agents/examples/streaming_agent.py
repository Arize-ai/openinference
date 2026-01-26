"""Streaming example using Strands with OpenInference instrumentation.

This example demonstrates streaming responses with proper trace capture
using the StrandsToOpenInferenceProcessor.
"""

import asyncio
import os

from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands.telemetry import StrandsTelemetry

from openinference.instrumentation.strands_agents import StrandsToOpenInferenceProcessor


@tool
async def search_database(query: str) -> dict:
    """Search a database for information.

    Args:
        query: The search query
    """
    await asyncio.sleep(0.5)  # Simulate async operation
    results = {
        "python": "Python is a high-level programming language known for readability",
        "javascript": "JavaScript is a scripting language primarily used for web development",
        "rust": "Rust is a systems programming language focused on safety and performance",
    }
    result = results.get(query.lower(), f"No results found for '{query}'")
    return {
        "status": "success",
        "content": [{"text": result}],
    }


async def main():
    """Run the streaming agent example."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please run: export OPENAI_API_KEY='your-api-key-here'")
        return

    print("=" * 60)
    print("Streaming Strands Agent with OpenInference Instrumentation")
    print("=" * 60)
    print()

    # Setup Strands' native telemetry
    print("📡 Setting up telemetry...")
    telemetry = StrandsTelemetry()
    telemetry.setup_otlp_exporter(endpoint="http://127.0.0.1:6006/v1/traces")

    # Add OpenInference processor
    telemetry.tracer_provider.add_span_processor(StrandsToOpenInferenceProcessor(debug=False))

    print("✅ Telemetry configured")
    print()

    # Create agent
    print("🤖 Creating agent...")
    model = OpenAIModel(model_id="gpt-4o-mini")
    agent = Agent(
        name="SearchAssistant",
        model=model,
        tools=[search_database],
        system_prompt="You are a helpful assistant that can search for information.",
    )
    print("✅ Agent created")
    print()

    # Run streaming query
    print("💬 Streaming query: 'Tell me about Python programming'")
    print("-" * 60)

    async for event in agent.stream_async("Tell me about Python programming"):
        if "data" in event:
            print(event["data"], end="", flush=True)

    print()
    print("-" * 60)
    print()
    print("✨ Done! Check Phoenix at http://127.0.0.1:6006 to see traces")


if __name__ == "__main__":
    asyncio.run(main())
