"""Regression tests for span attributes the OpenAI Agents SDK leaves off spans.

The SDK does not populate:

* input/output on agent, task, turn, and handoff spans (including the trace
  root shown in the trace list),
* name/input/output on handoff (TOOL) spans,
* tool.parameters / tool.description on function (TOOL) spans,

and it stamps llm.system on every span kind. The processor now fills the first
three gaps by deriving values from child LLM/response spans, scopes llm.system to
LLM spans only, and adds agent.name to agent spans (matching other instrumentors).

Covers both the Responses API (ResponseSpanData) and Chat Completions
(GenerationSpanData) code paths.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

try:
    # TaskSpanData / TurnSpanData were added in a later agents release; on older
    # pinned versions the import raises ImportError and the module is skipped.
    from agents.tracing.span_data import (  # type: ignore[attr-defined]
        AgentSpanData,
        FunctionSpanData,
        GenerationSpanData,
        HandoffSpanData,
        ResponseSpanData,
        SpanData,
        TaskSpanData,
        TurnSpanData,
    )
    from openai.types.responses import (
        FunctionTool,
        Response,
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )
except ImportError:
    pytest.skip(
        "agents package incompatible with current OpenAI SDK version", allow_module_level=True
    )


from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.openai_agents._processor import OpenInferenceTracingProcessor


def _function_tool() -> "FunctionTool":
    return FunctionTool(
        type="function",
        name="get_weather",
        description="Get the current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        strict=True,
    )


def _text_response(text: str) -> "Response":
    return Response(
        id="resp-text",
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
        tools=[_function_tool()],
    )


def _tool_call_response() -> "Response":
    return Response(
        id="resp-toolcall",
        created_at=0.0,
        model="gpt-4o-mini",
        object="response",
        output=[
            ResponseFunctionToolCall(
                type="function_call",
                call_id="call-1",
                name="get_weather",
                arguments='{"city":"London"}',
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[_function_tool()],
    )


_TRACE_ID = "trace_abc"
_STARTED_AT = "2020-01-01T00:00:00+00:00"
_ENDED_AT = "2020-01-01T00:00:01+00:00"


class _FakeTrace:
    def __init__(self, trace_id: str, name: str) -> None:
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


def _make_processor() -> tuple[OpenInferenceTracingProcessor, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OpenInferenceTracingProcessor(provider.get_tracer(__name__)), exporter


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _by_name_kind(spans: list[ReadableSpan]) -> dict[tuple[str, Any], dict[str, Any]]:
    return {(s.name, _attrs(s).get("openinference.span.kind")): _attrs(s) for s in spans}


def _run_single_agent_trace() -> list[ReadableSpan]:
    """root -> task -> agent -> turn -> {generation (LLM), function (TOOL)}."""
    processor, exporter = _make_processor()
    trace = _FakeTrace(_TRACE_ID, "Agent workflow")
    task = _FakeSpan("task", None, TaskSpanData(name="Agent workflow"))
    agent = _FakeSpan("agent", "task", AgentSpanData(name="WeatherAgent"))
    turn = _FakeSpan("turn", "agent", TurnSpanData(turn=1, agent_name="WeatherAgent"))
    generation = _FakeSpan(
        "gen",
        "turn",
        GenerationSpanData(
            input=[{"role": "user", "content": "What's the weather in London?"}],
            output=[{"role": "assistant", "content": "It is 21C and sunny in London."}],
            model="gpt-4o-mini",
        ),
    )
    function = _FakeSpan(
        "func",
        "turn",
        FunctionSpanData(name="get_weather", input='{"city":"London"}', output="21C"),
    )
    # Seed a tool schema directly here; capture from response.tools is covered by
    # the Responses API tests below.
    processor._tool_schemas[(_TRACE_ID, "get_weather")] = (
        "Get the current weather for a city.",
        {"type": "object", "properties": {"city": {"type": "string"}}},
    )

    processor.on_trace_start(trace)  # type: ignore[arg-type]
    for span in (task, agent, turn, generation, function):
        processor.on_span_start(span)  # type: ignore[arg-type]
    for span in (generation, function, turn, agent, task):  # children before parents
        processor.on_span_end(span)  # type: ignore[arg-type]
    processor.on_trace_end(trace)  # type: ignore[arg-type]
    return list(exporter.get_finished_spans())


def test_parent_spans_receive_input_and_output() -> None:
    spans = _by_name_kind(_run_single_agent_trace())
    for key in [
        ("Agent workflow", "AGENT"),  # trace root — shown in the trace list
        ("Agent workflow", "CHAIN"),  # task span
        ("WeatherAgent", "AGENT"),
        ("turn", "CHAIN"),
    ]:
        attrs = spans[key]
        assert attrs.get("input.value") is not None, f"{key} missing input.value"
        assert attrs.get("output.value") is not None, f"{key} missing output.value"
        assert "What's the weather in London?" in str(attrs["input.value"])
        assert "It is 21C and sunny in London." in str(attrs["output.value"])


def test_leaf_llm_span_still_has_input_and_output() -> None:
    spans = _run_single_agent_trace()
    llm = next(s for s in spans if _attrs(s).get("openinference.span.kind") == "LLM")
    attrs = _attrs(llm)
    assert attrs.get("input.value") is not None
    assert attrs.get("output.value") is not None


def test_function_span_gets_tool_schema() -> None:
    spans = _run_single_agent_trace()
    tool = next(
        s
        for s in spans
        if _attrs(s).get("openinference.span.kind") == "TOOL" and s.name == "get_weather"
    )
    attrs = _attrs(tool)
    assert attrs.get("tool.name") == "get_weather"
    assert attrs.get("tool.description") == "Get the current weather for a city."
    assert attrs.get("tool.parameters") is not None
    assert "city" in str(attrs["tool.parameters"])


def test_llm_system_scoped_to_llm_spans() -> None:
    spans = _run_single_agent_trace()
    for s in spans:
        attrs = _attrs(s)
        kind = attrs.get("openinference.span.kind")
        if kind == "LLM":
            assert attrs.get("llm.system") is not None, f"LLM span {s.name} missing llm.system"
        else:
            assert attrs.get("llm.system") is None, f"{kind} span {s.name} has stray llm.system"


def test_handoff_span_is_populated() -> None:
    processor, exporter = _make_processor()
    trace = _FakeTrace(_TRACE_ID, "Agent workflow")
    task = _FakeSpan("task", None, TaskSpanData(name="Agent workflow"))
    agent = _FakeSpan("agent", "task", AgentSpanData(name="TriageAgent"))
    handoff = _FakeSpan(
        "handoff",
        "agent",
        HandoffSpanData(from_agent="TriageAgent", to_agent="TechSupportAgent"),
    )

    processor.on_trace_start(trace)  # type: ignore[arg-type]
    for span in (task, agent, handoff):
        processor.on_span_start(span)  # type: ignore[arg-type]
    for span in (handoff, agent, task):
        processor.on_span_end(span)  # type: ignore[arg-type]
    processor.on_trace_end(trace)  # type: ignore[arg-type]

    spans = exporter.get_finished_spans()
    handoff_span = next(
        s
        for s in spans
        if _attrs(s).get("openinference.span.kind") == "TOOL" and s.name.startswith("handoff to")
    )
    attrs = _attrs(handoff_span)
    # tool.name must match the tool call the model made (the SDK's default handoff
    # tool name), not the human-readable span name.
    assert attrs.get("tool.name") == "transfer_to_techsupportagent"
    assert attrs.get("input.value") == "TriageAgent"
    assert attrs.get("output.value") == "TechSupportAgent"


def test_turn_output_falls_back_to_tool_calls_when_no_text() -> None:
    """A turn whose LLM call only made a tool call (no assistant text) should still
    get an output (the tool call it requested), not an empty Output panel."""
    processor, exporter = _make_processor()
    trace = _FakeTrace(_TRACE_ID, "Agent workflow")
    task = _FakeSpan("task", None, TaskSpanData(name="Agent workflow"))
    agent = _FakeSpan("agent", "task", AgentSpanData(name="WeatherAgent"))
    turn = _FakeSpan("turn", "agent", TurnSpanData(turn=1, agent_name="WeatherAgent"))
    generation = _FakeSpan(
        "gen",
        "turn",
        GenerationSpanData(
            input=[{"role": "user", "content": "weather in London?"}],
            output=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "get_weather", "arguments": '{"city":"London"}'}}
                    ],
                }
            ],
            model="gpt-4o-mini",
        ),
    )

    processor.on_trace_start(trace)  # type: ignore[arg-type]
    for span in (task, agent, turn, generation):
        processor.on_span_start(span)  # type: ignore[arg-type]
    for span in (generation, turn, agent, task):
        processor.on_span_end(span)  # type: ignore[arg-type]
    processor.on_trace_end(trace)  # type: ignore[arg-type]

    turn_attrs = _by_name_kind(list(exporter.get_finished_spans()))[("turn", "CHAIN")]
    assert turn_attrs.get("output.value") is not None, "tool-call turn still missing output"
    assert "get_weather" in str(turn_attrs["output.value"])


def test_agent_span_has_agent_name() -> None:
    spans = _by_name_kind(_run_single_agent_trace())
    agent_attrs = spans[("WeatherAgent", "AGENT")]
    assert agent_attrs.get("agent.name") == "WeatherAgent"


# --- Responses API path (the default) -------------------------------------------------


def _run_responses_api_trace() -> list[ReadableSpan]:
    """A Responses-API trace: turn 1 makes a tool call (get_weather) then runs the
    tool; turn 2 returns the final text. Exercises response.output_text, the tool-call
    output fallback, and tool-schema capture from ``response.tools`` (not seeded).

    root -> task -> agent -> [turn1 -> {response(tool call), function}, turn2 -> response(text)]
    """
    processor, exporter = _make_processor()
    trace = _FakeTrace(_TRACE_ID, "Agent workflow")
    task = _FakeSpan("task", None, TaskSpanData(name="Agent workflow"))
    agent = _FakeSpan("agent", "task", AgentSpanData(name="WeatherAgent"))
    turn1 = _FakeSpan("turn1", "agent", TurnSpanData(turn=1, agent_name="WeatherAgent"))
    response1 = _FakeSpan(
        "resp1",
        "turn1",
        ResponseSpanData(
            response=_tool_call_response(),
            input=[{"role": "user", "content": "What's the weather in London?"}],
        ),
    )
    function = _FakeSpan(
        "func",
        "turn1",
        FunctionSpanData(name="get_weather", input='{"city":"London"}', output="21C"),
    )
    turn2 = _FakeSpan("turn2", "agent", TurnSpanData(turn=2, agent_name="WeatherAgent"))
    response2 = _FakeSpan(
        "resp2",
        "turn2",
        ResponseSpanData(
            response=_text_response("It is 21C and sunny in London."),
            input=[{"role": "user", "content": "What's the weather in London?"}],
        ),
    )

    processor.on_trace_start(trace)  # type: ignore[arg-type]
    for span in (task, agent, turn1, response1, function, turn2, response2):
        processor.on_span_start(span)  # type: ignore[arg-type]
    # response1 must end before function so the tool schema is captured first.
    for span in (response1, function, turn1, response2, turn2, agent, task):
        processor.on_span_end(span)  # type: ignore[arg-type]
    processor.on_trace_end(trace)  # type: ignore[arg-type]
    return list(exporter.get_finished_spans())


def test_responses_api_parents_get_text_output_not_raw_response() -> None:
    spans = _by_name_kind(_run_responses_api_trace())
    for key in [("Agent workflow", "AGENT"), ("WeatherAgent", "AGENT")]:
        attrs = spans[key]
        # Final answer text, not the serialized Response object.
        assert attrs.get("output.value") == "It is 21C and sunny in London."
        assert "created_at" not in str(attrs["output.value"])
        assert "What's the weather in London?" in str(attrs["input.value"])


def test_responses_api_tool_call_turn_output_is_the_tool_call() -> None:
    # Both turns share the span name "turn", so work off the raw span list.
    turns = [
        _attrs(s)
        for s in _run_responses_api_trace()
        if s.name == "turn" and _attrs(s).get("openinference.span.kind") == "CHAIN"
    ]
    assert len(turns) == 2
    outputs = [str(a.get("output.value")) for a in turns]
    # One turn made the tool call; the other produced the final text.
    assert any("get_weather" in o for o in outputs), "tool-call turn output missing"
    assert any("It is 21C and sunny in London." in o for o in outputs), "text turn output missing"


def test_responses_api_function_span_schema_captured_from_response_tools() -> None:
    # No schema is seeded here; it must be captured from response.tools.
    spans = _run_responses_api_trace()
    tool = next(
        s
        for s in spans
        if _attrs(s).get("openinference.span.kind") == "TOOL" and s.name == "get_weather"
    )
    attrs = _attrs(tool)
    assert attrs.get("tool.name") == "get_weather"
    assert attrs.get("tool.description") == "Get the current weather for a city."
    assert "city" in str(attrs.get("tool.parameters"))


def test_responses_api_llm_system_scoped_to_llm_spans() -> None:
    for s in _run_responses_api_trace():
        attrs = _attrs(s)
        kind = attrs.get("openinference.span.kind")
        if kind == "LLM":
            assert attrs.get("llm.system") is not None, f"LLM span {s.name} missing llm.system"
        else:
            assert attrs.get("llm.system") is None, f"{kind} span {s.name} has stray llm.system"
