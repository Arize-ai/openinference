"""
Tests for Team context isolation to ensure multiple team.run() calls
create separate top-level traces instead of nesting.

These tests should FAIL before the fix is applied and PASS after.
"""

from typing import Iterator, Sequence, Tuple

import pytest
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team import Team
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.agno import AgnoInstrumentor


@pytest.fixture
def instrumented_team() -> Iterator[Tuple[Team, InMemorySpanExporter]]:
    """Set up instrumentation and return a Team with an Agent."""
    # Create tracer provider with in-memory exporter
    tracer_provider = TracerProvider()
    exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Instrument
    instrumentor = AgnoInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)

    # Create agent
    agent = Agent(
        name="Test Agent",
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake-key-for-testing"),
        instructions="You are a test agent.",
    )

    # Create team
    team = Team(
        name="Test Team",
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake-key-for-testing"),
        members=[agent],
        instructions="Delegate questions to the agent.",
    )

    yield team, exporter

    # Cleanup
    instrumentor.uninstrument()
    exporter.clear()


def _get_team_spans(spans: Sequence[ReadableSpan]) -> list[ReadableSpan]:
    """Get all Team spans using attributes, not name strings."""
    team_spans: list[ReadableSpan] = []
    for span in spans:
        attributes = dict(span.attributes or {})
        # Identify Team spans by checking for agno.team.id or graph.node.name containing team
        if "agno.team.id" in attributes or (
            attributes.get("graph.node.name") and "Team" in str(attributes.get("graph.node.name"))
        ):
            team_spans.append(span)
    return team_spans


def test_multiple_team_runs_create_separate_traces(
    instrumented_team: Tuple[Team, InMemorySpanExporter],
) -> None:
    """
    Test that multiple team.run() calls create separate top-level traces,
    not nested traces.

    This is the key test that should FAIL before the fix.
    """
    team, exporter = instrumented_team

    # Run team three times
    try:
        team.run("First question")
    except Exception:
        pass  # Ignore failures due to fake API key

    try:
        team.run("Second question")
    except Exception:
        pass

    try:
        team.run("Third question")
    except Exception:
        pass

    # Get all spans
    spans = exporter.get_finished_spans()

    # Find all Team spans using attributes
    team_spans = _get_team_spans(spans)

    # Should have 3 team spans
    assert len(team_spans) >= 3, f"Expected at least 3 team spans, got {len(team_spans)}"

    # CRITICAL: Each team span should have NO parent span
    # (parent_id should be None or invalid)
    for team_span in team_spans:
        parent_span_context = team_span.parent
        assert parent_span_context is None or not parent_span_context.is_valid, (
            f"Team span has a parent span! "
            f"This means traces are nesting instead of being separate. "
            f"Parent: {parent_span_context}"
        )


def test_multiple_sequential_calls_without_nesting(
    instrumented_team: Tuple[Team, InMemorySpanExporter],
) -> None:
    """
    Test that the fix resolves nesting for sequential team calls.

    Before the fix, subsequent team.run() calls would nest under previous ones.
    After the fix, each should be a separate top-level trace.
    """
    team, exporter = instrumented_team

    # Make sequential calls
    try:
        team.run("Call 1")
    except Exception:
        pass

    try:
        team.run("Call 2")
    except Exception:
        pass

    spans = exporter.get_finished_spans()
    team_spans = _get_team_spans(spans)

    assert len(team_spans) >= 2, f"Expected at least 2 team spans, got {len(team_spans)}"

    # Verify neither span has the other as parent
    if len(team_spans) >= 2:
        span1, span2 = team_spans[0], team_spans[1]

        # Span 2 should NOT have Span 1 as parent
        if span2.parent and span2.parent.is_valid:
            span2_parent_id = span2.parent.span_id
            span1_id = span1.context.span_id
            assert span2_parent_id != span1_id, (
                "Second team call has first team call as parent - nesting detected!"
            )


def test_team_spans_have_required_attributes(
    instrumented_team: Tuple[Team, InMemorySpanExporter],
) -> None:
    """
    Test that team spans have required OpenInference attributes.

    This ensures our fix doesn't break attribute collection.
    """
    team, exporter = instrumented_team

    try:
        team.run("Question for team")
    except Exception:
        pass

    spans = exporter.get_finished_spans()

    # Find team spans
    team_spans = _get_team_spans(spans)

    assert len(team_spans) >= 1, "No team spans found"

    # Check required attributes on first team span
    team_span = team_spans[0]
    attributes = dict(team_span.attributes or {})

    assert "openinference.span.kind" in attributes, "Missing openinference.span.kind"
    assert attributes["openinference.span.kind"] == "AGENT"
    assert "graph.node.id" in attributes, "Team span missing graph.node.id attribute"


@pytest.mark.asyncio
async def test_multiple_async_team_runs_create_separate_traces(
    instrumented_team: Tuple[Team, InMemorySpanExporter],
) -> None:
    """
    Test that multiple team.arun() calls create separate top-level traces.

    Same as the sync test but for async execution.
    """
    team, exporter = instrumented_team

    # Run team three times asynchronously
    try:
        await team.arun("First async question")
    except Exception:
        pass

    try:
        await team.arun("Second async question")
    except Exception:
        pass

    try:
        await team.arun("Third async question")
    except Exception:
        pass

    spans = exporter.get_finished_spans()
    team_spans = _get_team_spans(spans)

    assert len(team_spans) >= 3, f"Expected at least 3 async team spans, got {len(team_spans)}"

    # Each team span should have NO parent
    for team_span in team_spans:
        parent_span_context = team_span.parent
        assert parent_span_context is None or not parent_span_context.is_valid, (
            f"Async team span has a parent span! Parent: {parent_span_context}"
        )


def test_sequential_team_runs_have_different_trace_ids(
    instrumented_team: Tuple[Team, InMemorySpanExporter],
) -> None:
    """
    Test that sequential team.run() calls have different trace IDs,
    confirming they are truly separate traces.
    """
    team, exporter = instrumented_team

    # Run team multiple times
    try:
        team.run("Team question 1")
    except Exception:
        pass

    try:
        team.run("Team question 2")
    except Exception:
        pass

    try:
        team.run("Team question 3")
    except Exception:
        pass

    spans = exporter.get_finished_spans()

    # Find all team spans
    team_spans = _get_team_spans(spans)

    assert len(team_spans) >= 3, f"Expected at least 3 team spans, got {len(team_spans)}"

    # Collect all unique trace IDs
    trace_ids = {format(s.context.trace_id, "032x") for s in team_spans}

    # All team spans should have DIFFERENT trace IDs (separate traces)
    assert len(trace_ids) == len(team_spans), (
        f"Expected {len(team_spans)} unique trace IDs, but got {len(trace_ids)}. "
        f"This means some team runs are sharing the same trace (nesting)!"
    )


def test_team_context_cleanup_between_runs(
    instrumented_team: Tuple[Team, InMemorySpanExporter],
) -> None:
    """
    Test that context is properly cleaned up between team runs.

    This verifies that internal context state doesn't leak between calls.
    """
    team, exporter = instrumented_team

    # First run
    try:
        team.run("First run")
    except Exception:
        pass

    spans_after_first = exporter.get_finished_spans()
    team_spans_first = _get_team_spans(spans_after_first)
    assert len(team_spans_first) >= 1, "No team span found after first run"
    first_team_span = team_spans_first[0]

    # Clear for second run
    exporter.clear()

    # Second run
    try:
        team.run("Second run")
    except Exception:
        pass

    spans_after_second = exporter.get_finished_spans()
    team_spans_second = _get_team_spans(spans_after_second)
    assert len(team_spans_second) >= 1, "No team span found after second run"
    second_team_span = team_spans_second[0]

    # Second span should NOT reference first span as parent
    if second_team_span.parent is not None and second_team_span.parent.is_valid:
        second_parent_id = second_team_span.parent.span_id
        first_span_id = first_team_span.context.span_id

        assert second_parent_id != first_span_id, (
            "Second team run incorrectly has first team run as parent! "
            "Context is leaking between calls."
        )


@pytest.mark.asyncio
async def test_mixed_sync_async_team_runs_create_separate_traces(
    instrumented_team: Tuple[Team, InMemorySpanExporter],
) -> None:
    """
    Test that mixing sync and async team runs still creates separate traces.
    """
    team, exporter = instrumented_team

    # Mix sync and async calls
    try:
        team.run("Sync question 1")
    except Exception:
        pass

    try:
        await team.arun("Async question")
    except Exception:
        pass

    try:
        team.run("Sync question 2")
    except Exception:
        pass

    spans = exporter.get_finished_spans()
    team_spans = _get_team_spans(spans)

    assert len(team_spans) >= 3, f"Expected at least 3 team spans, got {len(team_spans)}"

    # All should be separate traces (no parent)
    for team_span in team_spans:
        parent_span_context = team_span.parent
        assert parent_span_context is None or not parent_span_context.is_valid, (
            "Mixed sync/async team span has a parent span!"
        )
