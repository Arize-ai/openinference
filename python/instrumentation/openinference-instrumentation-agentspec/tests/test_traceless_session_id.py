"""Regression tests for session.id serialization on traceless spans.

A span opened without an active AgentSpec Trace has ``span._trace is None``.
The conversion helper must omit ``session.id`` in that case: serializing a
``None`` value produces the literal string ``"null"``, which would group
unrelated traceless spans into a single bogus session.
"""

from typing import Any

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pyagentspec.agent import Agent
from pyagentspec.llms import OpenAiConfig
from pyagentspec.tracing.spans import AgentExecutionSpan
from pyagentspec.tracing.trace import Trace as AgentSpecTrace
from pyagentspec.tracing.trace import get_trace

from openinference.instrumentation.agentspec._openinference_spanprocessor import (
    OpenInferenceSpanProcessor,
)
from openinference.semconv.trace import SpanAttributes


def _make_agent() -> Agent:
    return Agent(
        name="assistant",
        llm_config=OpenAiConfig(name="openai-llm", model_id="non-existent-model"),
        system_prompt="You are a helpful agent",
    )


def _make_processor(exporter: InMemorySpanExporter) -> OpenInferenceSpanProcessor:
    return OpenInferenceSpanProcessor(span_exporter=exporter, use_batch_span_processor=False)


def _exported_attrs(finished_span: Any) -> dict[str, Any]:
    return dict(finished_span.attributes or {})


def test_traceless_span_does_not_export_session_id() -> None:
    """A span with no active Trace must not export a ``session.id`` attribute."""
    assert get_trace() is None  # precondition: no active trace in this test context

    exporter = InMemorySpanExporter()
    processor = _make_processor(exporter)
    span = AgentExecutionSpan(agent=_make_agent())
    assert span._trace is None  # precondition: genuinely traceless

    processor.on_start(span)
    processor.on_end(span)

    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    attrs = _exported_attrs(finished[0])
    assert SpanAttributes.SESSION_ID not in attrs
    # Guard against the exact regression: the None used to serialize to "null".
    assert attrs.get(SpanAttributes.SESSION_ID) != "null"


def test_traced_span_still_exports_session_id() -> None:
    """With an active Trace, ``session.id`` must still equal the trace id."""
    assert get_trace() is None  # precondition: no active trace in this test context

    exporter = InMemorySpanExporter()
    processor = _make_processor(exporter)

    with AgentSpecTrace(span_processors=[processor]) as trace:
        with AgentExecutionSpan(agent=_make_agent()):
            pass

    agent_spans = [s for s in exporter.get_finished_spans() if s.name == "AgentExecutionSpan"]
    assert len(agent_spans) == 1
    attrs = _exported_attrs(agent_spans[0])
    assert attrs.get(SpanAttributes.SESSION_ID) == trace.id
