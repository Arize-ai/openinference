"""Tests for Team context isolation - sequential runs should create separate traces."""

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
    tracer_provider = TracerProvider()
    exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    instrumentor = AgnoInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)

    agent = Agent(
        name="Test Agent",
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake-key-for-testing"),
        instructions="You are a test agent.",
    )

    team = Team(
        name="Test Team",
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake-key-for-testing"),
        members=[agent],
        instructions="Delegate questions to the agent.",
    )

    yield team, exporter

    instrumentor.uninstrument()
    exporter.clear()


def _get_team_spans(spans: Sequence[ReadableSpan]) -> list[ReadableSpan]:
    """Filter Team spans by attributes."""
    team_spans: list[ReadableSpan] = []
    for span in spans:
        attributes = dict(span.attributes or {})
        if "agno.team.id" in attributes or (
            attributes.get("graph.node.name") and "Team" in str(attributes.get("graph.node.name"))
        ):
            team_spans.append(span)
    return team_spans


def test_multiple_team_runs_create_separate_traces(
    instrumented_team: Tuple[Team, InMemorySpanExporter],
) -> None:
    team, exporter = instrumented_team

    try:
        team.run("First question")
    except Exception:
        pass

    try:
        team.run("Second question")
    except Exception:
        pass

    try:
        team.run("Third question")
    except Exception:
        pass

    spans = exporter.get_finished_spans()
    team_spans = _get_team_spans(spans)

    assert len(team_spans) >= 3, f"Expected at least 3 team spans, got {len(team_spans)}"

    for team_span in team_spans:
        parent_span_context = team_span.parent
        assert parent_span_context is None or not parent_span_context.is_valid, (
            "Team span has parent - traces are nesting instead of being separate"
        )


def test_multiple_sequential_calls_without_nesting(
    instrumented_team: Tuple[Team, InMemorySpanExporter],
) -> None:
    team, exporter = instrumented_team

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

    if len(team_spans) >= 2:
        span1, span2 = team_spans[0], team_spans[1]

        if span2.parent and span2.parent.is_valid:
            span2_parent_id = span2.parent.span_id
            span1_id = span1.context.span_id
            assert span2_parent_id != span1_id, "Second team run nested under first"


def test_team_spans_have_required_attributes(
    instrumented_team: Tuple[Team, InMemorySpanExporter],
) -> None:
    team, exporter = instrumented_team

    try:
        team.run("Question for team")
    except Exception:
        pass

    spans = exporter.get_finished_spans()
    team_spans = _get_team_spans(spans)

    assert len(team_spans) >= 1, "No team spans found"

    team_span = team_spans[0]
    attributes = dict(team_span.attributes or {})

    assert "openinference.span.kind" in attributes
    assert attributes["openinference.span.kind"] == "AGENT"
    assert "graph.node.id" in attributes


@pytest.mark.asyncio
async def test_multiple_async_team_runs_create_separate_traces(
    instrumented_team: Tuple[Team, InMemorySpanExporter],
) -> None:
    team, exporter = instrumented_team

    try:
        await team.arun("First async question")  # type: ignore[misc]
    except Exception:
        pass

    try:
        await team.arun("Second async question")  # type: ignore[misc]
    except Exception:
        pass

    try:
        await team.arun("Third async question")  # type: ignore[misc]
    except Exception:
        pass

    spans = exporter.get_finished_spans()
    team_spans = _get_team_spans(spans)

    assert len(team_spans) >= 3, f"Expected at least 3 async team spans, got {len(team_spans)}"

    for team_span in team_spans:
        parent_span_context = team_span.parent
        assert parent_span_context is None or not parent_span_context.is_valid, (
            "Async team span has parent - traces are nesting"
        )


def test_sequential_team_runs_have_different_trace_ids(
    instrumented_team: Tuple[Team, InMemorySpanExporter],
) -> None:
    team, exporter = instrumented_team

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
    team_spans = _get_team_spans(spans)

    assert len(team_spans) >= 3, f"Expected at least 3 team spans, got {len(team_spans)}"

    trace_ids = {format(s.context.trace_id, "032x") for s in team_spans}

    assert len(trace_ids) == len(team_spans), (
        f"Expected {len(team_spans)} unique trace IDs, got {len(trace_ids)} - teams sharing traces"
    )


def test_team_context_cleanup_between_runs(
    instrumented_team: Tuple[Team, InMemorySpanExporter],
) -> None:
    team, exporter = instrumented_team

    try:
        team.run("First run")
    except Exception:
        pass

    spans_after_first = exporter.get_finished_spans()
    team_spans_first = _get_team_spans(spans_after_first)
    assert len(team_spans_first) >= 1, "No team span found after first run"
    first_team_span = team_spans_first[0]

    exporter.clear()

    try:
        team.run("Second run")
    except Exception:
        pass

    spans_after_second = exporter.get_finished_spans()
    team_spans_second = _get_team_spans(spans_after_second)
    assert len(team_spans_second) >= 1, "No team span found after second run"
    second_team_span = team_spans_second[0]

    if second_team_span.parent is not None and second_team_span.parent.is_valid:
        second_parent_id = second_team_span.parent.span_id
        first_span_id = first_team_span.context.span_id

        assert second_parent_id != first_span_id, "Context leaking between runs"


@pytest.mark.asyncio
async def test_mixed_sync_async_team_runs_create_separate_traces(
    instrumented_team: Tuple[Team, InMemorySpanExporter],
) -> None:
    team, exporter = instrumented_team

    try:
        team.run("Sync question 1")
    except Exception:
        pass

    try:
        await team.arun("Async question")  # type: ignore[misc]
    except Exception:
        pass

    try:
        team.run("Sync question 2")
    except Exception:
        pass

    spans = exporter.get_finished_spans()
    team_spans = _get_team_spans(spans)

    assert len(team_spans) >= 3, f"Expected at least 3 team spans, got {len(team_spans)}"

    for team_span in team_spans:
        parent_span_context = team_span.parent
        assert parent_span_context is None or not parent_span_context.is_valid, (
            "Mixed sync/async runs are nesting"
        )
