from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import using_attributes
from openinference.instrumentation.crewai._event_assembler import (
    _FINISHED_CONTEXT_CACHE_SIZE,
    CrewAIEventAssembler,
    _SpanEndSpec,
    _SpanStartSpec,
)
from openinference.semconv.trace import OpenInferenceSpanKindValues

from ._span_helpers import (
    INPUT_MIME_TYPE,
    LLM_TOKEN_COUNT_TOTAL,
    OPENINFERENCE_SPAN_KIND,
    OUTPUT_MIME_TYPE,
    OUTPUT_VALUE,
    SESSION_ID,
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
    start_event = _event("llm-start", type="llm_call_started")
    finish_event = _end_event("llm-start", type="llm_call_completed")

    assembler.end_span(
        finish_event,
        _SpanEndSpec(output="cached answer", attributes={LLM_TOKEN_COUNT_TOTAL: 17}),
    )
    assembler.start_span(
        start_event,
        _SpanStartSpec(
            name="gpt-4.llm_call",
            span_kind=OpenInferenceSpanKindValues.LLM,
            attributes={"llm.model_name": "gpt-4"},
        ),
    )

    spans = _finished_spans(in_memory_span_exporter)
    assert len(spans) == 1

    span = spans[0]
    attributes: dict[str, Any] = dict(span.attributes or {})
    assert span.name == "gpt-4.llm_call"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert attributes.pop("llm.model_name") == "gpt-4"
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 17
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


def test_context_attributes_are_captured_at_start_time(
    assembler: CrewAIEventAssembler,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    start_event = _event("agent-start")
    end_event = _end_event("agent-start")

    with using_attributes(session_id="captured-session"):
        assembler.start_span(
            start_event,
            _SpanStartSpec(name="Research.execute", span_kind=OpenInferenceSpanKindValues.AGENT),
        )

    assembler.end_span(end_event, _SpanEndSpec(output="done"))

    spans = _finished_spans(in_memory_span_exporter)
    assert len(spans) == 1
    attributes: dict[str, Any] = dict(spans[0].attributes or {})
    assert attributes.pop(SESSION_ID) == "captured-session"


def test_orphan_state_is_bounded(assembler: CrewAIEventAssembler) -> None:
    for index in range(_FINISHED_CONTEXT_CACHE_SIZE + 10):
        assembler.end_span(
            _end_event(f"missing-start-{index}"),
            _SpanEndSpec(output=f"end-{index}"),
        )
        assembler.start_span(
            _event(f"pending-child-{index}", parent_event_id=f"missing-parent-{index}"),
            _SpanStartSpec(
                name=f"pending-{index}",
                span_kind=OpenInferenceSpanKindValues.TOOL,
            ),
        )
        assembler.close_scope(_end_event(f"missing-scope-{index}"))

    assert len(assembler._deferred_ends) == _FINISHED_CONTEXT_CACHE_SIZE
    assert len(assembler._pending_starts) == _FINISHED_CONTEXT_CACHE_SIZE
    assert len(assembler._closed_transparent_scopes) == _FINISHED_CONTEXT_CACHE_SIZE


def test_pending_start_chain_drains_iteratively(assembler: CrewAIEventAssembler) -> None:
    depth = 1_100
    for index in range(depth, 0, -1):
        assembler.start_span(
            _event(f"node-{index}", parent_event_id=f"node-{index - 1}"),
            _SpanStartSpec(name=f"node-{index}", span_kind=OpenInferenceSpanKindValues.CHAIN),
        )

    assembler.start_span(
        _event("node-0"),
        _SpanStartSpec(name="node-0", span_kind=OpenInferenceSpanKindValues.CHAIN),
    )

    assert len(assembler._spans) == _FINISHED_CONTEXT_CACHE_SIZE + 1
    assert "node-0" in assembler._spans
    assert f"node-{_FINISHED_CONTEXT_CACHE_SIZE}" in assembler._spans
    assert f"node-{_FINISHED_CONTEXT_CACHE_SIZE + 1}" not in assembler._spans
