from contextlib import nullcontext
from typing import Any, ContextManager, cast

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import REDACTED_VALUE, OITracer, TraceConfig, using_session
from openinference.instrumentation.livekit_agents import LiveKitAgentsInstrumentor
from openinference.instrumentation.livekit_agents._processor import LiveKitTurnProcessor


def _setup(config: TraceConfig) -> tuple[TracerProvider, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    oi_tracer = OITracer(provider.get_tracer("openinference-livekit-agents"), config)
    provider.add_span_processor(LiveKitTurnProcessor(oi_tracer, config))
    return provider, exporter


def _drive_cascaded_turn(provider: TracerProvider, session_id: str | None = None) -> None:
    tracer = provider.get_tracer("livekit-agents")
    session_context = cast(
        ContextManager[Any],
        using_session(session_id) if session_id is not None else nullcontext(),
    )
    with session_context:
        session = tracer.start_span("agent_session", start_time=50)
        livekit_context = trace_api.set_span_in_context(session)
        user = tracer.start_span(
            "user_turn",
            context=livekit_context,
            attributes={"lk.pii.user_transcript": "hello"},
            start_time=100,
        )
        user.end(end_time=200)

        agent = tracer.start_span(
            "agent_turn",
            context=livekit_context,
            attributes={
                "lk.speech_id": "sp-1",
                "gen_ai.conversation.id": "RM_room",
            },
            start_time=300,
        )
        agent_context = trace_api.set_span_in_context(agent)

        llm_node = tracer.start_span("llm_node", context=agent_context, start_time=400)
        llm_node_context = trace_api.set_span_in_context(llm_node)
        llm_request = tracer.start_span(
            "llm_request",
            context=llm_node_context,
            attributes={
                "gen_ai.request.model": "gpt",
                "gen_ai.input.messages": '[{"role":"user","content":"hello"}]',
                "gen_ai.output.messages": '[{"role":"assistant","content":"hi"}]',
                "gen_ai.usage.input_tokens": 3,
                "gen_ai.usage.output_tokens": 5,
            },
            start_time=500,
        )
        llm_request.end(end_time=600)
        llm_node.set_attribute("lk.response.ttft", 0.25)
        llm_node.end(end_time=700)

        tool = tracer.start_span(
            "function_tool",
            context=agent_context,
            attributes={
                "gen_ai.tool.name": "lookup",
                "gen_ai.tool.call.result": "ok",
            },
            start_time=550,
        )
        tool.end(end_time=650)

        tts_node = tracer.start_span("tts_node", context=agent_context, start_time=800)
        tts_request = tracer.start_span(
            "tts_request",
            context=trace_api.set_span_in_context(tts_node),
            attributes={"lk.pii.input_text": "hi there"},
            start_time=810,
        )
        tts_request.end(end_time=900)
        tts_node.end(end_time=910)
        agent.end(end_time=1000)
        session.end(end_time=1100)


def _rendered_spans(exporter: InMemorySpanExporter) -> dict[str, Any]:
    names = {"conversation.turn", "user", "assistant", "lookup"}
    return {span.name: span for span in exporter.get_finished_spans() if span.name in names}


def _pop_attributes(span: Any, expected: dict[str, Any]) -> None:
    attributes = dict(span.attributes)
    for key, value in expected.items():
        assert attributes.pop(key) == value
    assert attributes == {}


def _assert_cascaded_attributes(
    spans: dict[str, Any],
    *,
    session_id: str,
    hide_inputs: bool = False,
) -> None:
    turn_input = (
        {"input.value": REDACTED_VALUE}
        if hide_inputs
        else {
            "input.value": "hello",
            "input.mime_type": "text/plain",
        }
    )
    assistant_input = (
        {"input.value": REDACTED_VALUE}
        if hide_inputs
        else {
            "input.value": '[{"role":"user","content":"hello"}]',
            "input.mime_type": "application/json",
        }
    )
    _pop_attributes(
        spans["conversation.turn"],
        {
            "openinference.span.kind": "AUDIO",
            "session.id": session_id,
            "llm.model_name": "gpt",
            "end_reason": "complete",
            **turn_input,
            "output.value": "hi there",
            "output.mime_type": "text/plain",
        },
    )
    user_expected = {
        "openinference.span.kind": "USER",
        "session.id": session_id,
    }
    if not hide_inputs:
        user_expected["input.audio.transcript"] = "hello"
    _pop_attributes(spans["user"], user_expected)
    _pop_attributes(
        spans["assistant"],
        {
            "openinference.span.kind": "LLM",
            "session.id": session_id,
            "llm.model_name": "gpt",
            **assistant_input,
            "output.value": "hi there",
            "output.mime_type": "text/plain",
            "output.audio.transcript": "hi there",
            "llm.token_count.prompt": 3,
            "llm.token_count.completion": 5,
            "time_to_first_token_ms": 250,
        },
    )
    _pop_attributes(
        spans["lookup"],
        {
            "openinference.span.kind": "TOOL",
            "session.id": session_id,
            "tool.name": "lookup",
            "output.value": "ok",
        },
    )


def test_cascaded_turn() -> None:
    provider, exporter = _setup(TraceConfig())
    _drive_cascaded_turn(provider)

    spans = _rendered_spans(exporter)
    assert set(spans) == {"conversation.turn", "user", "assistant", "lookup"}
    turn = spans["conversation.turn"]
    user = spans["user"]
    assistant = spans["assistant"]
    tool = spans["lookup"]

    assert turn.parent is None
    assert user.parent.span_id == turn.context.span_id
    assert assistant.parent.span_id == turn.context.span_id
    assert tool.parent.span_id == assistant.context.span_id

    _assert_cascaded_attributes(spans, session_id="RM_room")

    livekit_agent = next(
        span
        for span in exporter.get_finished_spans()
        if span.name == "agent_turn"
        and span.instrumentation_scope is not None
        and span.instrumentation_scope.name == "livekit-agents"
    )
    assert livekit_agent.attributes is not None
    assert livekit_agent.attributes["session.id"] == "RM_room"


def test_empty_agent_turn_renders_nothing() -> None:
    provider, exporter = _setup(TraceConfig())
    tracer = provider.get_tracer("livekit-agents")
    span = tracer.start_span("agent_turn")
    span.end()

    assert _rendered_spans(exporter) == {}


def test_caller_session_wins() -> None:
    provider, exporter = _setup(TraceConfig())
    _drive_cascaded_turn(provider, session_id="caller-session")

    spans = _rendered_spans(exporter)
    assert set(spans) == {"conversation.turn", "user", "assistant", "lookup"}
    _assert_cascaded_attributes(spans, session_id="caller-session")


def test_hide_inputs_redacts_turn_input_and_omits_audio_transcript() -> None:
    provider, exporter = _setup(TraceConfig(hide_inputs=True))
    _drive_cascaded_turn(provider)

    spans = _rendered_spans(exporter)
    assert set(spans) == {"conversation.turn", "user", "assistant", "lookup"}
    _assert_cascaded_attributes(spans, session_id="RM_room", hide_inputs=True)


def test_instrumentor_registers_and_disables_turn_processor() -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = LiveKitAgentsInstrumentor()

    instrumentor.instrument(tracer_provider=provider)
    _drive_cascaded_turn(provider)

    turns = [span for span in exporter.get_finished_spans() if span.name == "conversation.turn"]
    assert len(turns) == 1
    assert turns[0].attributes is not None
    assert turns[0].attributes["session.id"] == "RM_room"

    instrumentor.uninstrument()
    _drive_cascaded_turn(provider)

    turns = [span for span in exporter.get_finished_spans() if span.name == "conversation.turn"]
    assert len(turns) == 1
