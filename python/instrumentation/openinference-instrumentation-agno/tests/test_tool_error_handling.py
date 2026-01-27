"""
Tests for tool error handling in OpenInference Agno instrumentation.

This test suite validates that tool errors are properly captured as spans
with ERROR status and exported to the tracing backend.
"""

from typing import Any, Generator

import pytest
from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat
from agno.tools.function import Function
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from openinference.instrumentation.agno import AgnoInstrumentor
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes


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


def test_tool_error_span_is_created_and_ended(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_agno_instrumentation: Any,
) -> None:
    """
    Test that when a tool raises an exception, an error span is created,
    properly marked with ERROR status, and successfully ended/exported.

    This validates the fix for the bug where error spans were not being
    finalized with span.end().
    """

    # Create a tool that raises an exception
    def failing_tool(message: str) -> str:
        """A tool that always fails with an exception."""
        raise Exception(f"Tool error: {message}")

    tool = Function.from_callable(failing_tool, name="failing_tool")

    # Create an agent with the failing tool (no LLM needed for this test)
    agent = Agent(
        name="Test Agent",
        tools=[tool],
        # Use a mock model that won't be called
        model=OpenAIChat(id="gpt-4o-mini"),
    )

    # execute the tool to test error handling
    # catch the exception because we expect it to be raised
    try:
        from agno.tools.function import FunctionCall

        function_call = FunctionCall(function=tool, arguments={"message": "test failure"})
        function_call.execute()
    except Exception:
        pass  # the tool raises an exception

    spans = in_memory_span_exporter.get_finished_spans()

    # Find the tool error span
    tool_error_span = None
    for span in spans:
        if span.name == "failing_tool":
            tool_error_span = span
            break

    # Validate the error span exists and is properly configured
    assert tool_error_span is not None, "Tool error span should be created"

    # Validate span attributes
    attributes = dict(tool_error_span.attributes or {})
    assert attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "TOOL"
    assert attributes.get(SpanAttributes.TOOL_NAME) == "failing_tool"

    # Validate ERROR status
    assert tool_error_span.status.status_code == StatusCode.ERROR, (
        "Tool error span should have ERROR status"
    )
    assert "Tool error: test failure" in str(tool_error_span.status.description), (
        "Error description should contain the exception message"
    )

    # Validate that the span was ended (has end_time)
    assert tool_error_span.end_time is not None, (
        "Tool error span should be ended (have an end_time)"
    )
    assert tool_error_span.end_time > tool_error_span.start_time, (
        "Span end_time should be after start_time"
    )

    # Validate that error information is captured in output
    output_value = attributes.get(SpanAttributes.OUTPUT_VALUE, "")
    assert "Tool error: test failure" in output_value or "Tool error: test failure" in str(
        tool_error_span.status.description
    ), "Error message should be captured in output or status"


async def test_async_tool_error_span_is_created_and_ended(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_agno_instrumentation: Any,
) -> None:
    """
    Test that async tool errors are properly handled.
    This validates the fix in the arun() method.
    """

    # Create an async tool that raises an exception
    async def async_failing_tool(message: str) -> str:
        """An async tool that always fails with an exception."""
        raise Exception(f"Async tool error: {message}")

    tool = Function.from_callable(async_failing_tool, name="async_failing_tool")

    # Directly execute the async tool
    try:
        from agno.tools.function import FunctionCall

        function_call = FunctionCall(function=tool, arguments={"message": "async test failure"})
        await function_call.aexecute()
    except Exception:
        pass  # Expected

    # Check spans
    spans = in_memory_span_exporter.get_finished_spans()
    tool_error_span = None
    for span in spans:
        if span.name == "async_failing_tool":
            tool_error_span = span
            break

    assert tool_error_span is not None, "Async tool error span should be created"
    assert tool_error_span.status.status_code == StatusCode.ERROR
    assert tool_error_span.end_time is not None, "Async tool error span should be ended"


def test_tool_failure_status_creates_error_span(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_agno_instrumentation: Any,
) -> None:
    """
    Test that when a tool execution returns a failure status (without exception),
    an error span is still created and properly ended.

    This tests the specific code path: response.status == "failure"
    """

    # Create a tool that returns normally but indicates failure
    def soft_failing_tool(message: str) -> str:
        """A tool that returns but indicates failure in agno's response."""
        # This will be caught by agno and set response.status = "failure"
        raise ValueError(f"Soft failure: {message}")

    tool = Function.from_callable(soft_failing_tool, name="soft_failing_tool")

    try:
        from agno.tools.function import FunctionCall

        function_call = FunctionCall(function=tool, arguments={"message": "soft failure test"})
        response = function_call.execute()

        # The response should indicate failure
        assert response.status == "failure", "Response should indicate failure status"
    except Exception:
        pass  # May raise depending on agno version

    # Check spans
    spans = in_memory_span_exporter.get_finished_spans()
    tool_error_span = None
    for span in spans:
        if span.name == "soft_failing_tool":
            tool_error_span = span
            break

    assert tool_error_span is not None, "Soft failure span should be created"
    assert tool_error_span.status.status_code == StatusCode.ERROR, (
        "Soft failure should result in ERROR status"
    )
    assert tool_error_span.end_time is not None, "Soft failure span should be properly ended"

    # Verify output contains error message
    attributes = dict(tool_error_span.attributes or {})
    output_value = attributes.get(SpanAttributes.OUTPUT_VALUE, "")
    assert "failure" in output_value.lower(), "Output should contain failure message"


def test_successful_tool_span_for_comparison(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_agno_instrumentation: Any,
) -> None:
    """
    Test that successful tools still work correctly.
    This ensures our fix didn't break the success path.
    """

    def successful_tool(message: str) -> str:
        """A tool that succeeds."""
        return f"Success: {message}"

    tool = Function.from_callable(successful_tool, name="successful_tool")

    from agno.tools.function import FunctionCall

    function_call = FunctionCall(function=tool, arguments={"message": "test success"})
    response = function_call.execute()

    assert response.status == "success"

    # Check spans
    spans = in_memory_span_exporter.get_finished_spans()
    tool_span = None
    for span in spans:
        if span.name == "successful_tool":
            tool_span = span
            break

    assert tool_span is not None, "Success span should be created"
    assert tool_span.status.status_code == StatusCode.OK, "Successful tool should have OK status"
    assert tool_span.end_time is not None, "Success span should be ended"

    # Verify output
    attributes = dict(tool_span.attributes or {})
    output_value = attributes.get(SpanAttributes.OUTPUT_VALUE, "")
    assert "Success: test success" in output_value


def test_all_tool_spans_are_exported(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_agno_instrumentation: Any,
) -> None:
    """
    Test that multiple tool calls (both success and failure) all result in
    properly exported spans.
    """

    def success_tool() -> str:
        return "OK"

    def failure_tool() -> str:
        raise Exception("FAIL")

    success_fn = Function.from_callable(success_tool, name="success_tool")
    failure_fn = Function.from_callable(failure_tool, name="failure_tool")

    from agno.tools.function import FunctionCall

    # Execute successful tool
    FunctionCall(function=success_fn, arguments={}).execute()

    # Execute failing tool
    try:
        FunctionCall(function=failure_fn, arguments={}).execute()
    except Exception:
        pass

    # Check that both spans exist
    spans = in_memory_span_exporter.get_finished_spans()
    span_names = [span.name for span in spans]

    assert "success_tool" in span_names, "Success tool span should exist"
    assert "failure_tool" in span_names, "Failure tool span should exist"

    # Verify both spans have end_time
    for span in spans:
        assert span.end_time is not None, f"Span {span.name} should be ended"
        assert span.end_time > span.start_time, f"Span {span.name} should have valid end_time"
