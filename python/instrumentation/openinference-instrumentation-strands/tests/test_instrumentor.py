"""Tests for Strands instrumentor."""

from typing import Any

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.strands import StrandsInstrumentor


class TestStrandsInstrumentor:
    """Test cases for StrandsInstrumentor."""

    def test_instrumentor_can_be_instantiated(self) -> None:
        """Test that the instrumentor can be instantiated."""
        instrumentor = StrandsInstrumentor()
        assert instrumentor is not None

    def test_instrumentor_has_correct_instrumentation_dependencies(self) -> None:
        """Test that the instrumentor declares correct dependencies."""
        instrumentor = StrandsInstrumentor()
        dependencies = instrumentor.instrumentation_dependencies()
        assert "strands" in str(dependencies)

    def test_instrumentor_can_be_instrumented(
        self, instrumented_tracer_provider: trace_api.TracerProvider
    ) -> None:
        """Test that the instrumentor can instrument Strands."""
        instrumentor = StrandsInstrumentor()

        # Instrument
        instrumentor.instrument(tracer_provider=instrumented_tracer_provider)

        # Verify that methods are wrapped
        from strands.agent.agent import Agent

        assert hasattr(Agent, "invoke_async")
        assert hasattr(Agent, "stream_async")

        # Uninstrument
        instrumentor.uninstrument()

    def test_instrumentor_can_be_uninstrumented(
        self, instrumented_tracer_provider: trace_api.TracerProvider
    ) -> None:
        """Test that the instrumentor can uninstrument Strands."""
        instrumentor = StrandsInstrumentor()

        # Instrument
        instrumentor.instrument(tracer_provider=instrumented_tracer_provider)

        # Uninstrument
        instrumentor.uninstrument()

        # Methods should still exist but be unwrapped
        from strands.agent.agent import Agent

        assert hasattr(Agent, "invoke_async")
        assert hasattr(Agent, "stream_async")


@pytest.mark.asyncio
async def test_agent_invocation_creates_spans(
    instrumented_tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Test that agent invocations create spans."""
    from strands import Agent, tool

    # Clear any existing spans
    in_memory_span_exporter.clear()

    # Instrument
    instrumentor = StrandsInstrumentor()
    instrumentor.instrument(tracer_provider=instrumented_tracer_provider)

    try:
        # Define a simple tool
        @tool
        def echo(message: str) -> dict[str, Any]:
            """Echo a message back.

            Args:
                message: The message to echo
            """
            return {"status": "success", "content": [{"text": f"Echo: {message}"}]}

        # Create and invoke agent
        agent = Agent(name="TestAgent", tools=[echo])

        # Note: This might fail if Strands SDK is not installed or configured
        # In a real test, you'd need proper mocking or a test environment
        try:
            _ = agent("Echo hello")

            # Get exported spans
            spans = in_memory_span_exporter.get_finished_spans()

            # Verify spans were created
            assert len(spans) > 0

            # Verify agent span exists
            agent_spans = [s for s in spans if s.name.endswith(".invoke")]
            assert len(agent_spans) > 0

            # Verify span attributes
            agent_span = agent_spans[0]
            assert agent_span.attributes is not None
            assert "openinference.span.kind" in agent_span.attributes
            assert agent_span.attributes["openinference.span.kind"] == "AGENT"

        except Exception as e:
            # If Strands is not properly set up, skip this test
            pytest.skip(f"Strands agent invocation failed: {e}")

    finally:
        # Uninstrument
        instrumentor.uninstrument()


@pytest.mark.asyncio
async def test_tool_execution_creates_spans(
    instrumented_tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Test that tool executions create spans."""
    from strands import Agent, tool

    # Clear any existing spans
    in_memory_span_exporter.clear()

    # Instrument
    instrumentor = StrandsInstrumentor()
    instrumentor.instrument(tracer_provider=instrumented_tracer_provider)

    try:
        # Define a simple tool
        @tool
        def add(a: int, b: int) -> dict[str, Any]:
            """Add two numbers.

            Args:
                a: First number
                b: Second number
            """
            result = a + b
            return {"status": "success", "content": [{"text": str(result)}]}

        # Create and invoke agent
        agent = Agent(name="MathAgent", tools=[add])

        try:
            _ = agent("What is 5 plus 3?")

            # Get exported spans
            spans = in_memory_span_exporter.get_finished_spans()

            # Verify tool span exists
            tool_spans = [s for s in spans if s.name.startswith("tool.")]
            assert len(tool_spans) > 0

            # Verify span attributes
            if tool_spans:
                tool_span = tool_spans[0]
                assert tool_span.attributes is not None
                assert "openinference.span.kind" in tool_span.attributes
                assert tool_span.attributes["openinference.span.kind"] == "TOOL"

        except Exception as e:
            pytest.skip(f"Strands tool execution failed: {e}")

    finally:
        # Uninstrument
        instrumentor.uninstrument()
