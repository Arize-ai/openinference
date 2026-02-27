"""
CrewAI Agent - Standalone Kickoff Example

Demonstrates CrewAI instrumentation with a single agent invoked
directly via agent.kickoff(), outside of a Crew or Flow.
"""

import os

from crewai import Agent, LLM
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,  # type: ignore[import-not-found]
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.crewai import CrewAIInstrumentor

# OpenTelemetry Configuration
endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

# Disable CrewAI's built-in telemetry
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# Make sure to set the OPENAI_API_KEY environment variable
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"


def create_basic_agent() -> Agent:
    """
    Creates and returns an instance of a basic agent.

    Returns:
        Configured CrewAI agent ready for execution
    """
    llm = LLM(model="gpt-4o", temperature=0)
    return Agent(
        role="Helpful Assistant",
        goal="Answer questions clearly and concisely",
        backstory="You are a helpful assistant that provides clear answers.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def run_basic_agent() -> None:
    """
    Executes the basic agent and handles any runtime exceptions.
    """
    try:
        agent = create_basic_agent()
        query = "What is artificial intelligence?"
        print(f"Sending query to agent: {query}")
        result = agent.kickoff(query)
        print(f"\nAgent Response:\n{result}")
        print("\n" + "=" * 80)
        print("✅ Agent execution completed successfully.")
        print("=" * 80)
    except Exception as e:
        print(f"⚠️ Agent execution failed: {type(e).__name__}")


def main():
    """Run the CrewAI instrumentation demonstration."""
    print("=" * 80)
    print("CrewAI Instrumentation Demo - Standalone Agent Kickoff")
    print("=" * 80)
    print("Check Phoenix UI at http://localhost:6006 for trace visualization\n")
    run_basic_agent()


if __name__ == "__main__":
    # Instrument CrewAI with OpenTelemetry before running the agent
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    main()
