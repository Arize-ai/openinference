"""
CrewAI Flows - Basic Example

Demonstrates CrewAI instrumentation with a basic flow.
"""

import os
from typing import Optional

from crewai.flow.flow import Flow, listen, start
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,  # type: ignore[import-not-found]
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.crewai import CrewAIInstrumentor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

# Disable CrewAI's built-in telemetry
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# Make sure to set the OPENAI_API_KEY environment variable
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"


class BasicFlow(Flow):

    @start()
    def first_method(self):
        return "Output From First Method"

    @listen(first_method)
    def second_method(self, first_output):
        return f"Second Method Received: {first_output}"


def create_basic_flow(flow_name: Optional[str] = None) -> Flow:
    """
    Creates and returns an instance of a basic flow.

    Args:
        flow_name: Optional name to set for the flow

    Returns:
        Configured CrewAI flow ready for execution
    """
    flow = BasicFlow()
    if flow_name:
        flow.name = flow_name
    return flow


def run_basic_flow():
    """
    Executes the basic flow and handles any runtime exceptions.
    """
    try:
        flow = create_basic_flow("BasicFlowExample")
        flow.kickoff()
        print("✅ Flow execution completed successfully")
    except Exception as e:
        print(f"⚠️ Flow execution failed: {type(e).__name__}")


def main():
    """Run the CrewAI instrumentation demonstration."""
    print("CrewAI Instrumentation Demo - Basic Flow")
    print("Check Phoenix UI at http://localhost:6006 for trace visualization\n")

    # Run the flow
    run_basic_flow()


if __name__ == "__main__":
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    main()
