"""
CrewAI Flows - Basic Example
============================

A minimal example demonstrating how to define and run
a simple CrewAI Flow with OpenInference instrumentation.

This script:
- Initializes an OpenTelemetry tracer
- Defines a 2-step flow using @start and @listen
- Runs the flow and exports traces to Phoenix + Console
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

# OpenTelemetry Configuration
endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

# Disable CrewAI's built-in telemetry
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"


class BasicFlow(Flow):
    """
    A simple flow with two sequential steps.
    Demonstrates data passing between @start and @listen methods.
    """

    @start()
    def first_method(self) -> str:
        return "Output From First Method"

    @listen(first_method)
    def second_method(self, first_output: str) -> str:
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
        flow = create_basic_flow("Basic Flow Example")
        inputs = {"topic": "Machine Learning"}
        flow.kickoff(inputs=inputs)
        print("✅ Flow execution completed successfully.")
    except Exception as e:
        print(f"⚠️ Flow execution failed: {type(e).__name__}")


def main():
    """Run the CrewAI instrumentation demonstration."""
    print("CrewAI Instrumentation Demo - Basic Flow")
    print("Check Phoenix UI at http://localhost:6006 for trace visualization\n")
    run_basic_flow()


if __name__ == "__main__":
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    main()
