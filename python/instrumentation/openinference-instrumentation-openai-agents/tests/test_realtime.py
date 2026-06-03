"""Tests for OpenInference tracing of RealtimeAgent/RealtimeSession."""

from __future__ import annotations

import json
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

try:
    from agents.realtime.events import (
        RealtimeAgentEndEvent,
        RealtimeAgentStartEvent,
        RealtimeError,
        RealtimeGuardrailTripped,
        RealtimeHandoffEvent,
        RealtimeToolEnd,
        RealtimeToolStart,
    )
except ImportError:
    pytest.skip("openai-agents realtime module not available", allow_module_level=True)

from openinference.instrumentation import TraceConfig, using_attributes
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
TOOL_NAME = SpanAttributes.TOOL_NAME
INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
GRAPH_NODE_ID = SpanAttributes.GRAPH_NODE_ID
GRAPH_NODE_PARENT_ID = SpanAttributes.GRAPH_NODE_PARENT_ID
SESSION_ID = SpanAttributes.SESSION_ID
USER_ID = SpanAttributes.USER_ID

AGENT_KIND = OpenInferenceSpanKindValues.AGENT.value
TOOL_KIND = OpenInferenceSpanKindValues.TOOL.value
CHAIN_KIND = OpenInferenceSpanKindValues.CHAIN.value

JSON_MIME = "application/json"
TEXT_MIME = "text/plain"


# ---------------------------------------------------------------------------
# Helpers for building mock event objects
# ---------------------------------------------------------------------------


def _make_agent(name: str) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


def _make_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


def _make_event_info() -> MagicMock:
    return MagicMock()


def make_agent_start(name: str) -> RealtimeAgentStartEvent:
    return RealtimeAgentStartEvent(agent=_make_agent(name), info=_make_event_info())


def make_agent_end(name: str) -> RealtimeAgentEndEvent:
    return RealtimeAgentEndEvent(agent=_make_agent(name), info=_make_event_info())


def make_tool_start(agent_name: str, tool_name: str) -> RealtimeToolStart:
    return RealtimeToolStart(
        agent=_make_agent(agent_name),
        tool=_make_tool(tool_name),
        arguments="{}",
        info=_make_event_info(),
    )


def make_tool_end(agent_name: str, tool_name: str, output: Any) -> RealtimeToolEnd:
    return RealtimeToolEnd(
        agent=_make_agent(agent_name),
        tool=_make_tool(tool_name),
        arguments="{}",
        output=output,
        info=_make_event_info(),
    )


def make_handoff(from_name: str, to_name: str) -> RealtimeHandoffEvent:
    return RealtimeHandoffEvent(
        from_agent=_make_agent(from_name),
        to_agent=_make_agent(to_name),
        info=_make_event_info(),
    )


def make_error(msg: str) -> RealtimeError:
    return RealtimeError(error=msg, info=_make_event_info())


def make_guardrail_tripped(guardrail_name: str, message: str) -> RealtimeGuardrailTripped:
    guardrail = MagicMock()
    guardrail.name = guardrail_name
    result = MagicMock()
    result.guardrail = guardrail
    return RealtimeGuardrailTripped(
        guardrail_results=[result], message=message, info=_make_event_info()
    )


# ---------------------------------------------------------------------------
# Minimal mock session (proper class so dunder methods work correctly)
# ---------------------------------------------------------------------------


class _MockRealtimeSession:
    """Minimal stand-in for RealtimeSession that yields a fixed event list."""

    def __init__(self, agent_name: str, events: list[Any]) -> None:
        # New code reads _current_agent (matches the real RealtimeSession attribute name)
        self._current_agent = _make_agent(agent_name)
        self._events = events
        self._idx = 0

    async def __aenter__(self) -> _MockRealtimeSession:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    def __aiter__(self) -> _MockRealtimeSession:
        return self

    async def __anext__(self) -> Any:
        if self._idx >= len(self._events):
            raise StopAsyncIteration
        evt = self._events[self._idx]
        self._idx += 1
        return evt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> trace_api.TracerProvider:
    tp = trace_sdk.TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tp


@pytest.fixture
def instrument(tracer_provider: trace_api.TracerProvider) -> Iterator[None]:
    OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    OpenAIAgentsInstrumentor().uninstrument()


def _get_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return list(exporter.get_finished_spans())


# ---------------------------------------------------------------------------
# Helper to run a mock session through the instrumented RealtimeRunner
# ---------------------------------------------------------------------------


async def _run_session(events: list[Any], agent_name: str = "assistant") -> None:
    """Drive a mock realtime session through the instrumented wrapper."""
    from agents.realtime.runner import RealtimeRunner

    mock_session = _MockRealtimeSession(agent_name, events)

    with patch("agents.realtime.runner.RealtimeSession", return_value=mock_session):
        with patch("agents.realtime.runner.OpenAIRealtimeWebSocketModel"):
            runner = RealtimeRunner(starting_agent=_make_agent(agent_name))
            session = await runner.run()
            async with session:
                async for _ in session:
                    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_realtime_creates_agent_span(
    instrument: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    """Session enter/exit creates a single root AGENT span (no wrapper span)."""
    await _run_session(events=[], agent_name="my_agent")

    spans = _get_spans(in_memory_span_exporter)
    assert len(spans) == 1, f"Expected 1 span, got {[s.name for s in spans]}"

    root = spans[0]
    assert root.name == "Agent: my_agent"
    assert root.parent is None, "Agent span should be a root span"
    assert root.status.status_code.name == "OK"

    attributes = dict(root.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == AGENT_KIND
    assert attributes.pop(LLM_SYSTEM) == "openai"
    assert attributes.pop(GRAPH_NODE_ID) == "my_agent"
    assert not attributes


@pytest.mark.asyncio
async def test_realtime_agent_start_idempotent(
    instrument: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    """Multiple agent_start events for the same agent don't create duplicate spans."""
    # The SDK fires agent_start on every response.created (including after tool output)
    events = [
        make_agent_start("assistant"),
        make_agent_end("assistant"),
        make_agent_start("assistant"),  # second turn (e.g., after tool call)
        make_agent_end("assistant"),
    ]
    await _run_session(events, agent_name="assistant")

    spans = _get_spans(in_memory_span_exporter)
    agent_spans = [s for s in spans if s.name == "Agent: assistant"]
    assert len(agent_spans) == 1, f"Expected 1 agent span, got {len(agent_spans)}"


@pytest.mark.asyncio
async def test_realtime_tool_events_create_span(
    instrument: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    """ToolStart + ToolEnd events produce a child TOOL span with output."""
    events = [
        make_agent_start("assistant"),
        make_tool_start("assistant", "get_weather"),
        make_tool_end("assistant", "get_weather", "Sunny, 72°F"),
        make_agent_end("assistant"),
    ]
    await _run_session(events, agent_name="assistant")

    spans = _get_spans(in_memory_span_exporter)
    tool_span = next((s for s in spans if s.name == "Tool: get_weather"), None)
    assert tool_span is not None, f"Expected tool span, got {[s.name for s in spans]}"
    assert tool_span.status.status_code.name == "OK"

    attributes = dict(tool_span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL_KIND
    assert attributes.pop(LLM_SYSTEM) == "openai"
    assert attributes.pop(TOOL_NAME) == "get_weather"
    assert attributes.pop(OUTPUT_VALUE) == "Sunny, 72°F"
    assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT_MIME
    assert not attributes


@pytest.mark.asyncio
async def test_realtime_handoff_creates_span(
    instrument: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    """HandoffEvent produces a TOOL span with graph node attributes."""
    events = [
        make_agent_start("agent_a"),
        make_handoff("agent_a", "agent_b"),
        make_agent_end("agent_a"),
    ]
    await _run_session(events, agent_name="agent_a")

    spans = _get_spans(in_memory_span_exporter)
    handoff_span = next((s for s in spans if "Handoff:" in s.name), None)
    assert handoff_span is not None, f"Expected handoff span, got {[s.name for s in spans]}"
    assert handoff_span.name == "Handoff: agent_a -> agent_b"
    assert handoff_span.status.status_code.name == "OK"

    attributes = dict(handoff_span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL_KIND
    assert attributes.pop(LLM_SYSTEM) == "openai"
    assert attributes.pop(GRAPH_NODE_ID) == "agent_b"
    assert attributes.pop(GRAPH_NODE_PARENT_ID) == "agent_a"
    assert not attributes


@pytest.mark.asyncio
async def test_realtime_error_sets_error_status(
    instrument: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    """RealtimeError sets ERROR status on the current agent span."""
    events = [
        make_agent_start("assistant"),
        make_error("Something went wrong"),
        make_agent_end("assistant"),
    ]
    await _run_session(events, agent_name="assistant")

    spans = _get_spans(in_memory_span_exporter)
    agent_span = next((s for s in spans if s.name == "Agent: assistant"), None)
    assert agent_span is not None
    assert agent_span.status.status_code.name == "ERROR"
    assert "Something went wrong" in (agent_span.status.description or "")


@pytest.mark.asyncio
async def test_realtime_guardrail_creates_span(
    instrument: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    """GuardrailTripped event creates a CHAIN span with output."""
    events = [
        make_agent_start("assistant"),
        make_guardrail_tripped("content_policy", "Blocked content"),
        make_agent_end("assistant"),
    ]
    await _run_session(events, agent_name="assistant")

    spans = _get_spans(in_memory_span_exporter)
    guardrail_span = next((s for s in spans if "Guardrail:" in s.name), None)
    assert guardrail_span is not None, f"Expected guardrail span, got {[s.name for s in spans]}"
    assert guardrail_span.name == "Guardrail: content_policy"
    assert guardrail_span.status.status_code.name == "OK"

    attributes = dict(guardrail_span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN_KIND
    assert attributes.pop(LLM_SYSTEM) == "openai"
    assert attributes.pop(OUTPUT_VALUE) == "Blocked content"
    assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT_MIME
    assert not attributes


@pytest.mark.asyncio
async def test_realtime_span_hierarchy(
    instrument: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    """Verify correct parent-child span relationships: agent is root, tool is child of agent."""
    events = [
        make_agent_start("assistant"),
        make_tool_start("assistant", "lookup"),
        make_tool_end("assistant", "lookup", "result"),
        make_agent_end("assistant"),
    ]
    await _run_session(events, agent_name="assistant")

    spans = _get_spans(in_memory_span_exporter)
    agent_span = next((s for s in spans if s.name == "Agent: assistant"), None)
    tool_span = next((s for s in spans if s.name == "Tool: lookup"), None)

    assert agent_span is not None
    assert tool_span is not None

    # All spans share the same trace
    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 1

    # Agent span is the root
    assert agent_span.parent is None

    # Tool span's parent is the agent span
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == agent_span.context.span_id


@pytest.mark.asyncio
async def test_realtime_suppress_tracing(in_memory_span_exporter: InMemorySpanExporter) -> None:
    """No spans are created when instrumentation is suppressed."""
    import opentelemetry.context as context_api

    tp = trace_sdk.TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))

    OpenAIAgentsInstrumentor().instrument(tracer_provider=tp)
    try:
        events = [
            make_agent_start("assistant"),
            make_tool_start("assistant", "search"),
            make_tool_end("assistant", "search", "output"),
            make_agent_end("assistant"),
        ]

        suppress_key = context_api._SUPPRESS_INSTRUMENTATION_KEY
        token = context_api.attach(context_api.set_value(suppress_key, True))
        try:
            await _run_session(events, agent_name="assistant")
        finally:
            context_api.detach(token)

        realtime_spans = [
            s
            for s in _get_spans(in_memory_span_exporter)
            if any(k in s.name for k in ["Agent:", "Tool:", "Handoff:"])
        ]
        assert len(realtime_spans) == 0, (
            f"Expected no spans, got {[s.name for s in realtime_spans]}"
        )
    finally:
        OpenAIAgentsInstrumentor().uninstrument()


@pytest.mark.asyncio
async def test_realtime_multiple_agents_via_handoff(
    instrument: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    """Full flow: agent_a with tool → handoff → agent_b with tool. All in same trace."""
    events = [
        make_agent_start("agent_a"),
        make_tool_start("agent_a", "tool_a"),
        make_tool_end("agent_a", "tool_a", "result_a"),
        make_handoff("agent_a", "agent_b"),
        make_agent_end("agent_a"),
        make_agent_start("agent_b"),
        make_tool_start("agent_b", "tool_b"),
        make_tool_end("agent_b", "tool_b", "result_b"),
        make_agent_end("agent_b"),
    ]
    await _run_session(events, agent_name="agent_a")

    spans = _get_spans(in_memory_span_exporter)
    names = {s.name for s in spans}

    assert "Agent: agent_a" in names
    assert "Tool: tool_a" in names
    assert "Handoff: agent_a -> agent_b" in names
    assert "Agent: agent_b" in names
    assert "Tool: tool_b" in names

    # All spans share the same trace
    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 1

    tool_a = next(s for s in spans if s.name == "Tool: tool_a")
    tool_b = next(s for s in spans if s.name == "Tool: tool_b")

    attrs_a = dict(tool_a.attributes or {})
    assert attrs_a.pop(OPENINFERENCE_SPAN_KIND) == TOOL_KIND
    assert attrs_a.pop(LLM_SYSTEM) == "openai"
    assert attrs_a.pop(TOOL_NAME) == "tool_a"
    assert attrs_a.pop(OUTPUT_VALUE) == "result_a"
    assert attrs_a.pop(OUTPUT_MIME_TYPE) == TEXT_MIME
    assert not attrs_a

    attrs_b = dict(tool_b.attributes or {})
    assert attrs_b.pop(OPENINFERENCE_SPAN_KIND) == TOOL_KIND
    assert attrs_b.pop(LLM_SYSTEM) == "openai"
    assert attrs_b.pop(TOOL_NAME) == "tool_b"
    assert attrs_b.pop(OUTPUT_VALUE) == "result_b"
    assert attrs_b.pop(OUTPUT_MIME_TYPE) == TEXT_MIME
    assert not attrs_b


@pytest.mark.asyncio
async def test_realtime_tool_with_json_output(
    instrument: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    """Tool output that is a dict gets JSON-serialized with application/json MIME type."""
    events = [
        make_agent_start("assistant"),
        make_tool_start("assistant", "fetch_data"),
        make_tool_end("assistant", "fetch_data", {"key": "value", "count": 42}),
        make_agent_end("assistant"),
    ]
    await _run_session(events, agent_name="assistant")

    spans = _get_spans(in_memory_span_exporter)
    tool_span = next((s for s in spans if s.name == "Tool: fetch_data"), None)
    assert tool_span is not None

    attributes = dict(tool_span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL_KIND
    assert attributes.pop(LLM_SYSTEM) == "openai"
    assert attributes.pop(TOOL_NAME) == "fetch_data"
    output = attributes.pop(OUTPUT_VALUE)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON_MIME
    assert not attributes

    assert output is not None
    parsed = json.loads(output)  # type: ignore[arg-type]
    assert parsed == {"key": "value", "count": 42}


@pytest.mark.asyncio
async def test_realtime_concurrent_same_tool(
    instrument: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    """Two concurrent calls to the same tool name each get their own span."""
    events = [
        make_agent_start("assistant"),
        make_tool_start("assistant", "search"),
        make_tool_start("assistant", "search"),  # second concurrent call
        make_tool_end("assistant", "search", "result_2"),  # LIFO: ends second call
        make_tool_end("assistant", "search", "result_1"),  # ends first call
        make_agent_end("assistant"),
    ]
    await _run_session(events, agent_name="assistant")

    spans = _get_spans(in_memory_span_exporter)
    tool_spans = [s for s in spans if s.name == "Tool: search"]
    assert len(tool_spans) == 2, f"Expected 2 tool spans, got {len(tool_spans)}"

    outputs = {dict(s.attributes or {}).get(OUTPUT_VALUE) for s in tool_spans}
    assert outputs == {"result_1", "result_2"}


@pytest.mark.asyncio
async def test_realtime_uninstrument(in_memory_span_exporter: InMemorySpanExporter) -> None:
    """After uninstrument(), no realtime spans are created."""
    tp = trace_sdk.TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))

    OpenAIAgentsInstrumentor().instrument(tracer_provider=tp)
    OpenAIAgentsInstrumentor().uninstrument()

    events = [make_agent_start("assistant"), make_agent_end("assistant")]
    await _run_session(events, agent_name="assistant")

    realtime_spans = [
        s
        for s in _get_spans(in_memory_span_exporter)
        if any(k in s.name for k in ["Agent:", "Tool:"])
    ]
    assert len(realtime_spans) == 0, (
        f"Expected no spans after uninstrument, got {[s.name for s in realtime_spans]}"
    )


@pytest.mark.asyncio
async def test_realtime_no_events_single_agent_span(
    instrument: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    """Session with no events creates exactly one Agent span (the root)."""
    await _run_session(events=[], agent_name="assistant")

    spans = _get_spans(in_memory_span_exporter)
    assert len(spans) == 1, f"Expected 1 span, got {[s.name for s in spans]}"
    assert spans[0].name == "Agent: assistant"


@pytest.mark.asyncio
async def test_realtime_context_attribute_propagation(
    instrument: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    """Context attributes from using_attributes() propagate to the agent span."""
    events = [
        make_agent_start("assistant"),
        make_tool_start("assistant", "search"),
        make_tool_end("assistant", "search", "result"),
        make_agent_end("assistant"),
    ]
    with using_attributes(session_id="sess-123", user_id="user-456"):
        await _run_session(events, agent_name="assistant")

    spans = _get_spans(in_memory_span_exporter)
    agent = next(s for s in spans if s.name == "Agent: assistant")

    attrs = dict(agent.attributes or {})
    assert attrs.get(SESSION_ID) == "sess-123", "Agent span missing session_id"
    assert attrs.get(USER_ID) == "user-456", "Agent span missing user_id"


@pytest.mark.asyncio
async def test_realtime_hide_outputs(in_memory_span_exporter: InMemorySpanExporter) -> None:
    """hide_outputs=True in TraceConfig suppresses OUTPUT_VALUE on tool spans."""
    tp = trace_sdk.TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    OpenAIAgentsInstrumentor().instrument(tracer_provider=tp, config=TraceConfig(hide_outputs=True))
    try:
        events = [
            make_agent_start("assistant"),
            make_tool_start("assistant", "get_weather"),
            make_tool_end("assistant", "get_weather", "Sunny, 72°F"),
            make_agent_end("assistant"),
        ]
        await _run_session(events, agent_name="assistant")

        spans = _get_spans(in_memory_span_exporter)
        tool_span = next((s for s in spans if s.name == "Tool: get_weather"), None)
        assert tool_span is not None
        # OITracer replaces hidden output values with a redaction marker
        assert (tool_span.attributes or {}).get(OUTPUT_VALUE) == "__REDACTED__"
    finally:
        OpenAIAgentsInstrumentor().uninstrument()


@pytest.mark.asyncio
async def test_realtime_hide_inputs(in_memory_span_exporter: InMemorySpanExporter) -> None:
    """hide_inputs=True in TraceConfig suppresses INPUT_VALUE on agent spans."""
    tp = trace_sdk.TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    OpenAIAgentsInstrumentor().instrument(tracer_provider=tp, config=TraceConfig(hide_inputs=True))
    try:
        # Build a history_added event carrying user text so _self_last_user_text is set
        content = MagicMock()
        content.type = "input_text"
        content.text = "Hello, assistant"
        item = MagicMock()
        item.role = "user"
        item.content = [content]
        history_added = MagicMock()
        history_added.type = "history_added"
        history_added.item = item

        events = [
            make_agent_start("assistant"),
            history_added,
            make_agent_end("assistant"),
        ]
        await _run_session(events, agent_name="assistant")

        spans = _get_spans(in_memory_span_exporter)
        agent_span = next((s for s in spans if s.name == "Agent: assistant"), None)
        assert agent_span is not None
        # OITracer replaces hidden input values with a redaction marker
        assert (agent_span.attributes or {}).get(INPUT_VALUE) == "__REDACTED__"
    finally:
        OpenAIAgentsInstrumentor().uninstrument()
