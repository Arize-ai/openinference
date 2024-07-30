from typing import Any, Generator

import pytest
from crewai import Agent, Crew, Task
from crewai_tools import BaseTool  # type: ignore[import-untyped]
from openinference.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def setup_crewai_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    CrewAIInstrumentor().uninstrument()


class HelloWorldTool(BaseTool): # type: ignore[misc]
    name: str = "HelloWorldTool"
    description: str = "Tool that tells you helloworld for testing"

    def _run(self, argument: str) -> str:
        return "Hello World"


def test_crewai_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_crewai_instrumentation: Any,
) -> None:
    hello_world_tool = HelloWorldTool()
    greeter_agent = Agent(
        role="hello world greeter", goal="say hello world", tools=[hello_world_tool]
    )
    greeting_task = Task(
        description="get someone to greet the caller",
        expected_output="a greeting",
        agent=greeter_agent,
    )
    crew = Crew(
        agents=[greeter_agent],
        tasks=[
            greeting_task,
        ],
    )
    result = crew.kickoff()
    print(result)
