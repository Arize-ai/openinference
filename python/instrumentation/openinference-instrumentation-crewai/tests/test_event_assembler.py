from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.crewai._event_assembler import (
    CrewAIEventAssembler,
    _SpanEndSpec,
    _SpanStartSpec,
)
from openinference.semconv.trace import OpenInferenceSpanKindValues

from ._span_helpers import (
    INPUT_MIME_TYPE,
    OPENINFERENCE_SPAN_KIND,
    OUTPUT_MIME_TYPE,
    OUTPUT_VALUE,
    TEXT,
)

pytestmark = pytest.mark.no_autoinstrument


def _timestamp(offset_seconds: int = 0) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)


def _event(event_id: str, **kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(event_id=event_id, timestamp=_timestamp(), **kwargs)


def _end_event(started_event_id: str, **kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(started_event_id=started_event_id, timestamp=_timestamp(1), **kwargs)


def _finished_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return list(exporter.get_finished_spans())


def _span_by_name(spans: list[ReadableSpan], name: str) -> ReadableSpan:
    return next(span for span in spans if span.name == name)


@pytest.fixture()
def assembler(tracer_provider: trace_api.TracerProvider) -> CrewAIEventAssembler:
    tracer = trace_api.get_tracer(__name__, tracer_provider=tracer_provider)
    return CrewAIEventAssembler(tracer=tracer)


def test_deferred_end_preserves_completion_attributes(
    assembler: CrewAIEventAssembler,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    start_event = _event("tool-start", type="tool_usage_started")
    finish_event = _end_event("tool-start", type="tool_usage_finished")

    assembler.end_span(
        finish_event,
        _SpanEndSpec(output="cached answer", attributes={"tool.from_cache": True}),
    )
    assembler.start_span(
        start_event,
        _SpanStartSpec(
            name="search.run",
            span_kind=OpenInferenceSpanKindValues.TOOL,
            attributes={"tool.name": "search"},
        ),
    )

    spans = _finished_spans(in_memory_span_exporter)
    assert len(spans) == 1

    span = spans[0]
    attributes: dict[str, Any] = dict(span.attributes or {})
    assert span.name == "search.run"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.TOOL.value
    assert attributes.pop("tool.name") == "search"
    assert attributes.pop("tool.from_cache") is True
    assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
    assert attributes.pop(OUTPUT_VALUE) == "cached answer"
    assert attributes.pop(INPUT_MIME_TYPE, None) is None
    assert not attributes


def test_pending_child_start_attaches_to_late_parent(
    assembler: CrewAIEventAssembler,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    parent_event = _event("agent-start")
    child_event = _event("tool-start", parent_event_id="agent-start")

    assembler.start_span(
        child_event,
        _SpanStartSpec(
            name="search.run",
            span_kind=OpenInferenceSpanKindValues.TOOL,
            attributes={"tool.name": "search"},
        ),
    )
    assert not _finished_spans(in_memory_span_exporter)

    assembler.start_span(
        parent_event,
        _SpanStartSpec(
            name="Research.execute",
            span_kind=OpenInferenceSpanKindValues.AGENT,
            remember_as_agent=True,
        ),
    )
    assembler.end_span(_end_event("tool-start"), _SpanEndSpec(output="result"))
    assembler.end_span(_end_event("agent-start"), _SpanEndSpec(output="done"))

    spans = _finished_spans(in_memory_span_exporter)
    assert len(spans) == 2

    span_by_id = {span.context.span_id: span for span in spans}
    agent_span = _span_by_name(spans, "Research.execute")
    tool_span = _span_by_name(spans, "search.run")
    assert tool_span.parent is not None
    assert span_by_id[tool_span.parent.span_id].name == agent_span.name


def test_rootless_tool_prefers_active_task_scope_over_stale_agent_identity(
    assembler: CrewAIEventAssembler,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    shared_agent = SimpleNamespace(id="agent-shared", key="agent-key", role="Researcher")

    assembler.start_span(
        _event("crew-1"),
        _SpanStartSpec(name="crew-1", span_kind=OpenInferenceSpanKindValues.CHAIN),
    )
    assembler.start_span(
        _event(
            "agent-1",
            parent_event_id="crew-1",
            agent=shared_agent,
            task=SimpleNamespace(id="task-1"),
        ),
        _SpanStartSpec(
            name="Research.execute",
            span_kind=OpenInferenceSpanKindValues.AGENT,
            remember_as_agent=True,
        ),
    )
    assembler.end_span(_end_event("agent-1"), _SpanEndSpec(output="first run"))
    assembler.end_span(_end_event("crew-1"), _SpanEndSpec(output="first crew"))

    assembler.start_span(
        _event("crew-2"),
        _SpanStartSpec(name="crew-2", span_kind=OpenInferenceSpanKindValues.CHAIN),
    )
    assembler.open_scope(
        _event("task-2", parent_event_id="crew-2", task=SimpleNamespace(id="task-2"))
    )
    assembler.start_span(
        _event(
            "tool-2",
            type="tool_usage_started",
            task_id="task-2",
            agent_id="agent-shared",
            agent_key="agent-key",
            agent_role="Researcher",
        ),
        _SpanStartSpec(name="search.run", span_kind=OpenInferenceSpanKindValues.TOOL),
    )
    assembler.end_span(_end_event("tool-2"), _SpanEndSpec(output="second run"))
    assembler.close_scope(_end_event("task-2"))
    assembler.end_span(_end_event("crew-2"), _SpanEndSpec(output="second crew"))

    spans = _finished_spans(in_memory_span_exporter)
    assert len(spans) == 4

    crew_1_span = _span_by_name(spans, "crew-1")
    crew_2_span = _span_by_name(spans, "crew-2")
    tool_span = _span_by_name(spans, "search.run")
    assert tool_span.context.trace_id == crew_2_span.context.trace_id
    assert tool_span.context.trace_id != crew_1_span.context.trace_id
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == crew_2_span.context.span_id
