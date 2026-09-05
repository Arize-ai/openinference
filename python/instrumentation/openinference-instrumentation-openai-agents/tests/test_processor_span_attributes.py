"""Tests for span attributes the OpenAI Agents SDK leaves off spans.

The SDK stamps ``llm.system`` on every span kind, omits ``agent.name`` from agent
spans, omits the tool schema from function spans, and leaves handoff spans without
input or output. The processor fills those gaps using values the SDK does supply.

Note what is deliberately *not* done here: input/output are not inferred onto agent,
task, or turn spans from their child LLM spans. Those span data types carry no
input/output fields, so any value would be a guess -- see
``test_ancestor_spans_do_not_infer_input_output``.

The trace root does report the run's real input and output, but not from anything the
processor can see in span data: they are observed at the run boundary and handed over in
a ``ContextVar``. Only the processor's half is exercised here, by seeding the holder
directly; the plumbing that fills it is covered end to end in ``test_run_io.py``.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pytest
from agents.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    SpanData,
)
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionTool,
    Response,
    ResponseInputItemParam,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningItemParam,
)
from openai.types.responses.response_reasoning_item import Summary
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    suppress_tracing,
    using_metadata,
    using_session,
    using_tags,
    using_user,
)
from openinference.instrumentation.config import REDACTED_VALUE
from openinference.instrumentation.openai_agents._processor import OpenInferenceTracingProcessor
from openinference.instrumentation.openai_agents._run_io import RunIO, _run_io

_TRACE_ID = "trace_abc"
_STARTED_AT = "2020-01-01T00:00:00+00:00"
_ENDED_AT = "2020-01-01T00:00:01+00:00"

_TOOL_DESCRIPTION = "Get the current weather for a city."
_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}


class _FakeTrace:
    def __init__(self, trace_id: str = _TRACE_ID, name: str = "Agent workflow") -> None:
        self.trace_id = trace_id
        self.name = name


class _FakeSpan:
    def __init__(
        self,
        span_id: str,
        parent_id: Optional[str],
        span_data: SpanData,
        trace_id: str = _TRACE_ID,
    ) -> None:
        self.span_id = span_id
        self.parent_id = parent_id
        self.span_data = span_data
        self.trace_id = trace_id
        self.started_at = _STARTED_AT
        self.ended_at = _ENDED_AT
        self.error: Optional[dict[str, Any]] = None


def _function_tool(name: str = "get_weather") -> FunctionTool:
    return FunctionTool(
        type="function",
        name=name,
        description=_TOOL_DESCRIPTION,
        parameters=dict(_TOOL_PARAMETERS),
        strict=True,
    )


def _text_response(text: str = "It is 21C and sunny in London.", **kwargs: Any) -> Response:
    return Response(
        id="resp-1",
        created_at=0.0,
        model="gpt-4o-mini",
        object="response",
        output=[
            ResponseOutputMessage(
                id="m1",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=kwargs.pop("tools", [_function_tool()]),
        **kwargs,
    )


def _make_processor(
    config: Optional[TraceConfig] = None,
) -> tuple[OpenInferenceTracingProcessor, InMemorySpanExporter]:
    """Build a processor over an OITracer, matching how the instrumentor wires it up so
    that TraceConfig masking is exercised the same way it is in production.
    """
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OITracer(provider.get_tracer(__name__), config=config or TraceConfig())
    return OpenInferenceTracingProcessor(tracer), exporter  # type: ignore[arg-type]


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _run(
    processor: OpenInferenceTracingProcessor,
    trace: _FakeTrace,
    spans: list[_FakeSpan],
) -> None:
    """Drive ``spans`` through the processor, ending children before their parents."""
    processor.on_trace_start(trace)  # type: ignore[arg-type]
    for span in spans:
        processor.on_span_start(span)  # type: ignore[arg-type]
    for span in reversed(spans):
        processor.on_span_end(span)  # type: ignore[arg-type]
    processor.on_trace_end(trace)  # type: ignore[arg-type]


def _kinds(spans: list[ReadableSpan]) -> dict[str, dict[str, Any]]:
    """Map span kind -> attributes. Assumes one span per kind in the trace."""
    return {str(_attrs(s).get("openinference.span.kind")): _attrs(s) for s in spans}


def _one_of_kind(spans: list[ReadableSpan], kind: str, name: str) -> dict[str, Any]:
    matching = [
        _attrs(s)
        for s in spans
        if _attrs(s).get("openinference.span.kind") == kind and s.name == name
    ]
    assert len(matching) == 1, f"expected exactly one {kind} span named {name!r}"
    return matching[0]


# --- llm.system is scoped to LLM spans ----------------------------------------------


def test_llm_system_set_on_llm_spans() -> None:
    processor, exporter = _make_processor()
    response_span = _FakeSpan("resp", None, ResponseSpanData(response=_text_response(), input=None))
    generation_span = _FakeSpan(
        "gen", None, GenerationSpanData(input=None, output=None, model="gpt-4o-mini")
    )
    _run(processor, _FakeTrace(), [response_span, generation_span])

    llm_spans = [s for s in exporter.get_finished_spans() if _attrs(s).get("llm.system")]
    assert len(llm_spans) == 2
    for span in llm_spans:
        assert _attrs(span)["openinference.span.kind"] == "LLM"
        assert _attrs(span)["llm.system"] == "openai"


@pytest.mark.parametrize(
    "span_data,expected_kind",
    [
        (AgentSpanData(name="WeatherAgent"), "AGENT"),
        (FunctionSpanData(name="get_weather", input=None, output=None), "TOOL"),
        (HandoffSpanData(from_agent="Triage", to_agent="WeatherAgent"), "TOOL"),
        (CustomSpanData(name="custom", data={}), "CHAIN"),
        (GuardrailSpanData(name="length_check"), "GUARDRAIL"),
    ],
)
def test_llm_system_absent_on_non_llm_spans(span_data: SpanData, expected_kind: str) -> None:
    processor, exporter = _make_processor()
    _run(processor, _FakeTrace(), [_FakeSpan("s1", None, span_data)])

    non_root = [s for s in exporter.get_finished_spans() if s.name != "Agent workflow"]
    assert len(non_root) == 1
    attrs = _attrs(non_root[0])
    assert attrs["openinference.span.kind"] == expected_kind
    assert "llm.system" not in attrs


def test_llm_system_absent_on_trace_root_span() -> None:
    processor, exporter = _make_processor()
    _run(processor, _FakeTrace(), [])

    root = next(s for s in exporter.get_finished_spans() if s.name == "Agent workflow")
    assert _attrs(root)["openinference.span.kind"] == "AGENT"
    assert "llm.system" not in _attrs(root)


def test_response_spans_round_trip_reasoning_output_to_follow_up_input() -> None:
    processor, exporter = _make_processor()
    reasoning_id = "reason-123"
    reasoning_text = "The trains meet six hours after the first departure."
    encrypted_content = "encrypted-reasoning"

    first_response = _text_response("The answer is 6 hours.")
    first_response.output.insert(
        0,
        ResponseReasoningItem(
            id=reasoning_id,
            type="reasoning",
            summary=[Summary(type="summary_text", text=reasoning_text)],
            encrypted_content=encrypted_content,
        ),
    )
    follow_up_input: list[ResponseInputItemParam] = [
        EasyInputMessageParam(role="user", content="When do the trains meet?"),
        ResponseReasoningItemParam(
            id=reasoning_id,
            type="reasoning",
            summary=[{"type": "summary_text", "text": reasoning_text}],
            encrypted_content=encrypted_content,
        ),
        EasyInputMessageParam(role="assistant", content="The answer is 6 hours."),
        EasyInputMessageParam(role="user", content="Restate the answer in minutes."),
    ]
    spans = [
        _FakeSpan(
            "first-response",
            None,
            ResponseSpanData(response=first_response, input=follow_up_input[:1]),
        ),
        _FakeSpan(
            "follow-up-response",
            None,
            ResponseSpanData(
                response=_text_response("The answer is 360 minutes."),
                input=follow_up_input,
            ),
        ),
    ]
    _run(processor, _FakeTrace(), spans)

    llm_spans = [
        _attrs(span)
        for span in exporter.get_finished_spans()
        if _attrs(span).get("openinference.span.kind") == "LLM"
    ]
    first_attrs = next(
        attrs
        for attrs in llm_spans
        if attrs.get("llm.output_messages.0.message.contents.0.message_content.type") == "reasoning"
    )
    follow_up_attrs = next(
        attrs
        for attrs in llm_spans
        if attrs.get("llm.input_messages.2.message.contents.0.message_content.type") == "reasoning"
    )

    for attribute in ("type", "text", "id", "encrypted_content"):
        output_key = f"llm.output_messages.0.message.contents.0.message_content.{attribute}"
        input_key = f"llm.input_messages.2.message.contents.0.message_content.{attribute}"
        assert follow_up_attrs[input_key] == first_attrs[output_key]

    assert follow_up_attrs["llm.input_messages.4.message.role"] == "user"
    assert (
        follow_up_attrs["llm.input_messages.4.message.content"] == "Restate the answer in minutes."
    )


# --- agent.name on agent spans ------------------------------------------------------


def test_agent_span_records_agent_name() -> None:
    processor, exporter = _make_processor()
    _run(processor, _FakeTrace(), [_FakeSpan("a1", None, AgentSpanData(name="WeatherAgent"))])

    attrs = _one_of_kind(list(exporter.get_finished_spans()), "AGENT", "WeatherAgent")
    assert attrs.pop("openinference.span.kind") == "AGENT"
    assert attrs.pop("agent.name") == "WeatherAgent"
    # graph.node.id is pre-existing behaviour and must be preserved
    assert attrs.pop("graph.node.id") == "WeatherAgent"
    assert not attrs


# --- handoff span input/output ------------------------------------------------------


def test_handoff_span_records_from_and_to_agent() -> None:
    processor, exporter = _make_processor()
    _run(
        processor,
        _FakeTrace(),
        [_FakeSpan("h1", None, HandoffSpanData(from_agent="TriageAgent", to_agent="WeatherAgent"))],
    )

    attrs = _one_of_kind(list(exporter.get_finished_spans()), "TOOL", "handoff to WeatherAgent")
    assert attrs.pop("openinference.span.kind") == "TOOL"
    assert attrs.pop("input.value") == "TriageAgent"
    assert attrs.pop("output.value") == "WeatherAgent"
    assert not attrs


def test_handoff_span_does_not_guess_tool_name() -> None:
    """A handoff created with ``tool_name_override`` has a tool name that
    HandoffSpanData does not carry, so no tool.name is reported rather than a
    reconstructed one that could be wrong.
    """
    processor, exporter = _make_processor()
    _run(
        processor,
        _FakeTrace(),
        [_FakeSpan("h1", None, HandoffSpanData(from_agent="TriageAgent", to_agent="WeatherAgent"))],
    )

    attrs = _one_of_kind(list(exporter.get_finished_spans()), "TOOL", "handoff to WeatherAgent")
    assert "tool.name" not in attrs


def test_partial_handoff_span_records_only_known_agent() -> None:
    processor, exporter = _make_processor()
    _run(
        processor,
        _FakeTrace(),
        [_FakeSpan("h1", None, HandoffSpanData(from_agent="TriageAgent", to_agent=None))],
    )

    attrs = _one_of_kind(list(exporter.get_finished_spans()), "TOOL", "handoff")
    assert attrs["input.value"] == "TriageAgent"
    assert "output.value" not in attrs


@pytest.mark.parametrize(
    "config,masked_key,masked_value,kept_key,kept_value",
    [
        (TraceConfig(hide_inputs=True), "input.value", REDACTED_VALUE, "output.value", "Weather"),
        (TraceConfig(hide_outputs=True), "output.value", REDACTED_VALUE, "input.value", "Triage"),
    ],
)
def test_handoff_span_io_respects_trace_config_masking(
    config: TraceConfig,
    masked_key: str,
    masked_value: str,
    kept_key: str,
    kept_value: str,
) -> None:
    """The agent names recorded on handoff spans are input/output values, so they must
    be maskable like any other I/O.
    """
    processor, exporter = _make_processor(config)
    _run(
        processor,
        _FakeTrace(),
        [_FakeSpan("h1", None, HandoffSpanData(from_agent="Triage", to_agent="Weather"))],
    )

    attrs = _one_of_kind(list(exporter.get_finished_spans()), "TOOL", "handoff to Weather")
    assert attrs[masked_key] == masked_value
    assert attrs[kept_key] == kept_value


def test_handoff_still_populates_graph_node_parent_id() -> None:
    """Pre-existing graph edge behaviour must survive the added input/output."""
    processor, exporter = _make_processor()
    handoff = _FakeSpan("h1", None, HandoffSpanData(from_agent="TriageAgent", to_agent="Weather"))
    agent = _FakeSpan("a1", None, AgentSpanData(name="Weather"))

    processor.on_trace_start(_FakeTrace())  # type: ignore[arg-type]
    for span in (handoff, agent):
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]

    attrs = _one_of_kind(list(exporter.get_finished_spans()), "AGENT", "Weather")
    assert attrs["graph.node.parent_id"] == "TriageAgent"


# --- agent/task/turn spans keep empty I/O ------------------------------------------


def test_ancestor_spans_do_not_infer_input_output() -> None:
    """Agent spans and the trace root must not borrow I/O from child LLM spans.

    The SDK's agent span data has no input/output fields, so a value derived from
    child spans would be inferred rather than observed. This asserts the processor
    leaves them unset.
    """
    processor, exporter = _make_processor()
    trace = _FakeTrace()
    agent = _FakeSpan("a1", None, AgentSpanData(name="WeatherAgent"))
    response = _FakeSpan(
        "r1",
        "a1",
        ResponseSpanData(
            response=_text_response("It is 21C and sunny in London."),
            input=[{"role": "user", "content": "What's the weather in London?"}],
        ),
    )
    _run(processor, trace, [agent, response])

    spans = list(exporter.get_finished_spans())
    # The trace root carries nothing but its span kind.
    assert _one_of_kind(spans, "AGENT", "Agent workflow") == {"openinference.span.kind": "AGENT"}
    # The agent span carries only its own identity, no borrowed I/O.
    assert _one_of_kind(spans, "AGENT", "WeatherAgent") == {
        "openinference.span.kind": "AGENT",
        "agent.name": "WeatherAgent",
        "graph.node.id": "WeatherAgent",
    }

    # The LLM span itself still carries the real I/O.
    llm = _kinds(spans)["LLM"]
    assert "What's the weather in London?" in str(llm["input.value"])
    assert "It is 21C and sunny in London." in str(llm["output.value"])


def test_trace_root_records_observed_run_io_and_agent_spans_still_do_not() -> None:
    """The trace root's I/O comes from the run boundary, never from its children.

    The pairing with the test above is the point: the holder is seeded with values that
    appear nowhere in the span data below, so the root can only have got them from the
    run boundary -- and the agent span is still expected to carry neither. How the values
    reach the holder is covered end to end in ``test_run_io.py``.
    """
    processor, exporter = _make_processor()
    agent = _FakeSpan("a1", None, AgentSpanData(name="WeatherAgent"))
    response = _FakeSpan(
        "r1",
        "a1",
        ResponseSpanData(
            response=_text_response("It is 21C and sunny in London."),
            input=[{"role": "user", "content": "What's the weather in London?"}],
        ),
    )

    token = _run_io.set(RunIO(input="the real question", output="the real answer"))
    try:
        _run(processor, _FakeTrace(), [agent, response])
    finally:
        _run_io.reset(token)

    spans = list(exporter.get_finished_spans())
    assert _one_of_kind(spans, "AGENT", "Agent workflow") == {
        "openinference.span.kind": "AGENT",
        "input.value": "the real question",
        "input.mime_type": "text/plain",
        "output.value": "the real answer",
        "output.mime_type": "text/plain",
    }
    # Unchanged: the agent span borrows nothing, including from the holder.
    assert _one_of_kind(spans, "AGENT", "WeatherAgent") == {
        "openinference.span.kind": "AGENT",
        "agent.name": "WeatherAgent",
        "graph.node.id": "WeatherAgent",
    }


def test_trace_root_reports_no_output_when_the_run_did_not_finish() -> None:
    """A run that raised or ran out of turns never reports a final output."""
    processor, exporter = _make_processor()

    token = _run_io.set(RunIO(input="a question"))
    try:
        _run(processor, _FakeTrace(), [])
    finally:
        _run_io.reset(token)

    attrs = _one_of_kind(list(exporter.get_finished_spans()), "AGENT", "Agent workflow")
    assert attrs.pop("openinference.span.kind") == "AGENT"
    assert attrs.pop("input.value") == "a question"
    assert attrs.pop("input.mime_type") == "text/plain"
    assert not attrs


# --- suppress tracing ---------------------------------------------------------------


def test_no_spans_when_tracing_suppressed() -> None:
    """Nothing is exported while suppressed, so none of the added attributes appear.

    Suppression has to be active across the whole run: the tracer reads the key when
    the span starts, which is where the processor asks for it.
    """
    processor, exporter = _make_processor()
    spans = [
        _FakeSpan("a1", None, AgentSpanData(name="WeatherAgent")),
        _FakeSpan("h1", "a1", HandoffSpanData(from_agent="Triage", to_agent="Weather")),
        _FakeSpan("f1", "a1", FunctionSpanData(name="get_weather", input=None, output=None)),
    ]

    with suppress_tracing():
        _run(processor, _FakeTrace(), spans)

    assert exporter.get_finished_spans() == ()


# --- context attribute propagation --------------------------------------------------


@pytest.mark.parametrize(
    "span_data,kind,span_name,expected",
    [
        pytest.param(
            AgentSpanData(name="WeatherAgent"),
            "AGENT",
            "WeatherAgent",
            {"agent.name": "WeatherAgent"},
            id="agent-name",
        ),
        pytest.param(
            HandoffSpanData(from_agent="Triage", to_agent="Weather"),
            "TOOL",
            "handoff to Weather",
            {"input.value": "Triage", "output.value": "Weather"},
            id="handoff-io",
        ),
    ],
)
def test_context_attributes_propagate_to_new_span_attributes(
    span_data: SpanData,
    kind: str,
    span_name: str,
    expected: dict[str, Any],
) -> None:
    """Context attributes must land on the spans carrying the added attributes."""
    processor, exporter = _make_processor()

    with (
        using_session("s-1"),
        using_user("u-1"),
        using_metadata({"k": "v"}),
        using_tags(["t1", "t2"]),
    ):
        _run(processor, _FakeTrace(), [_FakeSpan("s1", None, span_data)])

    attrs = _one_of_kind(list(exporter.get_finished_spans()), kind, span_name)
    assert attrs["session.id"] == "s-1"
    assert attrs["user.id"] == "u-1"
    assert json.loads(str(attrs["metadata"])) == {"k": "v"}
    assert list(attrs["tag.tags"]) == ["t1", "t2"]
    # The added attributes must survive alongside the context attributes.
    for key, value in expected.items():
        assert attrs[key] == value
