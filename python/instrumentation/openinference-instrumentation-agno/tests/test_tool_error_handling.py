"""
Tests for tool error handling in OpenInference Agno instrumentation.

This test suite validates that tool errors are properly captured as spans
with ERROR status and exported to the tracing backend.
"""

from typing import Any, Generator

import pytest
import vcr  # type: ignore
from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat
from agno.tools.function import Function
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from openinference.instrumentation.agno import AgnoInstrumentor
from openinference.semconv.trace import SpanAttributes

test_vcr = vcr.VCR(
    serializer="yaml",
    cassette_library_dir="tests/openinference/instrumentation/agno/fixtures/",
    record_mode="never",
    match_on=["uri", "method"],
)


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
def setup_agno_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    AgnoInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    AgnoInstrumentor().uninstrument()


def test_agent_with_failing_tool_integration(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_agno_instrumentation: Any,
) -> None:
    """
    Integration test: Verify that when an agent uses a tool that fails,
    the error is properly captured with ERROR status in the full workflow.

    This uses a real agent.run() call with a VCR cassette.
    """
    with test_vcr.use_cassette(
        "agent_with_failing_tool.yaml", filter_headers=["authorization", "X-API-KEY"]
    ):
        import os

        os.environ["OPENAI_API_KEY"] = "fake_key"

        # Create a tool that will fail
        def unreliable_search(query: str) -> str:
            """A search tool that fails."""
            raise RuntimeError(f"Search service unavailable for: {query}")

        failing_tool = Function.from_callable(unreliable_search, name="unreliable_search")

        agent = Agent(
            name="Search Agent",
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[failing_tool],
            user_id="test_user",
        )

        # Run the agent - it will attempt to use the tool
        try:
            agent.run("Search for Python tutorials", session_id="test_session")
        except Exception:
            # Agent may fail if tool is critical
            pass

    spans = in_memory_span_exporter.get_finished_spans()

    # Find tool span
    tool_span = None
    for span in spans:
        if span.name == "unreliable_search":
            tool_span = span
            break

    # Verify tool was called and error handling worked
    assert tool_span is not None, "Failing tool span should be created"
    assert tool_span.status.status_code == StatusCode.ERROR, "Failing tool should have ERROR status"
    assert tool_span.end_time is not None, "Failing tool span should be ended"
    assert "Search service unavailable" in str(tool_span.status.description)

    attributes = dict(tool_span.attributes or {})
    assert attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "TOOL"
    assert attributes.get(SpanAttributes.TOOL_NAME) == "unreliable_search"


def test_agent_with_mixed_tool_results_integration(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_agno_instrumentation: Any,
) -> None:
    """
    Integration test: Agent with multiple tools where some succeed and some fail.
    Verifies that both success and error spans are properly captured.
    """
    with test_vcr.use_cassette(
        "agent_with_mixed_tools.yaml", filter_headers=["authorization", "X-API-KEY"]
    ):
        import os

        os.environ["OPENAI_API_KEY"] = "fake_key"

        # Create working and failing tools
        def working_calculator(a: int, b: int) -> int:
            """A calculator that works."""
            return a + b

        def broken_weather(city: str) -> str:
            """A weather tool that's broken."""
            raise ConnectionError(f"Weather service down for: {city}")

        working_tool = Function.from_callable(working_calculator, name="calculator")
        broken_tool = Function.from_callable(broken_weather, name="weather_lookup")

        agent = Agent(
            name="Assistant Agent",
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[working_tool, broken_tool],
        )

        try:
            agent.run("Calculate 2+2 and check weather in NYC", session_id="mixed_test")
        except Exception:
            pass

    spans = in_memory_span_exporter.get_finished_spans()

    # Check for tool spans
    calculator_span = None
    weather_span = None

    for span in spans:
        if span.name == "calculator":
            calculator_span = span
        elif span.name == "weather_lookup":
            weather_span = span

    # Assert both spans were created
    assert calculator_span is not None, "Calculator span should be created"
    assert weather_span is not None, "Weather span should be created"

    # Verify calculator succeeded
    assert calculator_span.status.status_code == StatusCode.OK
    assert calculator_span.end_time is not None

    # Verify weather failed
    assert weather_span.status.status_code == StatusCode.ERROR
    assert weather_span.end_time is not None
    assert "Weather service down" in str(weather_span.status.description)
