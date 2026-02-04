import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.claude_code._span_manager import SpanManager


@pytest.fixture
def span_manager(tracer_provider):
    """Create span manager with test tracer."""
    from openinference.instrumentation import OITracer, TraceConfig

    tracer = OITracer(
        tracer_provider.get_tracer(__name__),
        config=TraceConfig(),
    )
    return SpanManager(tracer)


def test_create_root_agent_span(span_manager, in_memory_span_exporter):
    """Test creating root AGENT span."""
    session_id = "test-session-123"

    span = span_manager.start_agent_span(
        name="Claude Code Query Session",
        session_id=session_id,
    )

    assert span is not None
    span_manager.end_span(span)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "Claude Code Query Session"


def test_create_child_llm_span(span_manager, in_memory_span_exporter):
    """Test creating child LLM span under agent."""
    root_span = span_manager.start_agent_span(
        name="Claude Code Query Session",
        session_id="test-session",
    )

    llm_span = span_manager.start_llm_span(
        name="Agent Turn 1",
        parent_span=root_span,
    )

    assert llm_span is not None
    span_manager.end_span(llm_span)
    span_manager.end_span(root_span)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    # Verify parent-child relationship
    assert spans[0].name == "Agent Turn 1"
    assert spans[1].name == "Claude Code Query Session"
    assert spans[0].parent.span_id == spans[1].context.span_id
