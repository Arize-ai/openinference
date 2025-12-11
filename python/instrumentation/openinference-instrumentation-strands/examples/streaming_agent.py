"""
Example of using Strands instrumentation with streaming.

This example demonstrates how to instrument a Strands agent
with streaming responses using OpenAI.

Before running:
    export OPENAI_API_KEY='your-api-key-here'
"""

import asyncio

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation import using_attributes
from openinference.instrumentation.strands import StrandsInstrumentor
from strands import Agent, tool
from strands.models.openai import OpenAIModel

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

StrandsInstrumentor().instrument(tracer_provider=tracer_provider)


@tool
async def search_database(query: str) -> dict:
    """Search a database for information.

    Args:
        query: The search query
    """
    await asyncio.sleep(0.5)
    results = {
        "python": "Python is a high-level programming language",
        "javascript": "JavaScript is a scripting language for the web",
    }
    result = results.get(query.lower(), f"No results found for '{query}'")
    return {"status": "success", "content": [{"text": result}]}


async def main():
    model = OpenAIModel(model_id="gpt-4o-mini")
    agent = Agent(
        name="Search Assistant",
        model=model,
        tools=[search_database],
        system_prompt="You are a helpful search assistant.",
    )

    with using_attributes(
        session_id="my-test-session",
        user_id="my-test-user",
        metadata={"example": "streaming", "framework": "strands"},
        tags=["search", "streaming"],
    ):
        async for event in agent.stream_async("Tell me about Python"):
            if "data" in event:
                print(event["data"], end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())

