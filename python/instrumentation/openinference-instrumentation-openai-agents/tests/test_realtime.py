from __future__ import annotations

import asyncio
from typing import Any

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import OITracer, TraceConfig, suppress_tracing, using_session
from openinference.instrumentation.openai_agents import _realtime
from openinference.instrumentation.openai_agents._realtime import (
    _AUDIO_KIND,
    _USER_KIND,
    _dispatch_event,
    _dispatch_raw,
    _RealtimeSessionState,
    _session_states,
    make_close_wrapper,
    make_realtime_wrapper,
    make_send_audio_wrapper,
    pcm16_to_wav_data_uri,
    truncate_audio_data_uri,
)
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes


def _state(
    tracer_provider: trace_api.TracerProvider,
    config: TraceConfig | None = None,
) -> _RealtimeSessionState:
    cfg = config or TraceConfig()
    tracer = OITracer(
        trace_api.get_tracer(__name__, tracer_provider=tracer_provider),
        config=cfg,
    )
    return _RealtimeSessionState(tracer, cfg)


def _drive_audio_turn(state: _RealtimeSessionState) -> None:
    """Drive a single complete turn with both user and assistant audio."""
    state.on_speech_started("item-1")
    state.on_send_audio(b"\x00\x01" * 256)
    state.on_user_audio_committed("item-1")
    state.on_user_transcript_completed("item-1", "Hi.")
    state.on_response_created("response-1", "gpt-realtime")
    state.on_audio_delta("response-1", b"\x02\x03" * 256)
    state.on_asst_transcript_done("response-1", "Hello!")
    state.on_response_done("response-1", None, "gpt-realtime")
    state.on_session_close()


def _spans_by_kind(in_memory_span_exporter: InMemorySpanExporter) -> dict[str, list[Any]]:
    spans: dict[str, list[Any]] = {}
    for span in in_memory_span_exporter.get_finished_spans():
        attributes = span.attributes or {}
        kind = attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        if isinstance(kind, str):
            spans.setdefault(kind, []).append(span)
    return spans


def _raw_event(raw: dict[str, Any]) -> Any:
    class _Inner:
        data = raw

    class _Event:
        data = _Inner()

    return _Event()


def test_multiple_assistant_responses_stay_under_one_audio_turn(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    state = _state(tracer_provider)

    state.on_speech_started("item-1")
    state.on_user_transcript_completed("item-1", "What is the weather?")
    state.on_response_created("response-1", "gpt-realtime")
    state.on_asst_transcript_done("response-1", "It is sunny.")
    state.on_response_done("response-1", None, "gpt-realtime")
    state.on_response_created("response-2", "gpt-realtime")
    state.on_asst_transcript_done("response-2", "Bring sunglasses.")
    state.on_response_done("response-2", None, "gpt-realtime")
    state.on_session_close()

    spans = _spans_by_kind(in_memory_span_exporter)
    audio_spans = spans[_AUDIO_KIND]
    user_spans = spans[_USER_KIND]
    llm_spans = spans[OpenInferenceSpanKindValues.LLM.value]

    assert len(audio_spans) == 1
    assert len(user_spans) == 1
    assert len(llm_spans) == 2
    audio_span_id = audio_spans[0].context.span_id
    assert all(span.parent is not None for span in user_spans + llm_spans)
    assert all(span.parent.span_id == audio_span_id for span in user_spans + llm_spans)
    assert audio_spans[0].attributes[SpanAttributes.INPUT_VALUE] == "What is the weather?"
    assert audio_spans[0].attributes[SpanAttributes.OUTPUT_VALUE] == (
        "It is sunny.\nBring sunglasses."
    )
    assert audio_spans[0].attributes["end_reason"] == "complete"


def test_new_user_input_after_assistant_output_starts_new_audio_turn(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    state = _state(tracer_provider)

    state.on_speech_started("item-1")
    state.on_user_transcript_completed("item-1", "What is the weather?")
    state.on_response_created("response-1", None)
    state.on_asst_transcript_done("response-1", "It is sunny.")
    state.on_response_done("response-1", None, None)
    state.on_speech_started("item-2")
    state.on_user_transcript_completed("item-2", "What about tomorrow?")
    state.on_response_created("response-2", None)
    state.on_asst_transcript_done("response-2", "Cloudy.")
    state.on_response_done("response-2", None, None)
    state.on_session_close()

    spans = _spans_by_kind(in_memory_span_exporter)
    audio_spans = spans[_AUDIO_KIND]
    user_spans = spans[_USER_KIND]
    llm_spans = spans[OpenInferenceSpanKindValues.LLM.value]

    assert len(audio_spans) == 2
    assert len(user_spans) == 2
    assert len(llm_spans) == 2
    assert {span.attributes[SpanAttributes.INPUT_VALUE] for span in audio_spans} == {
        "What is the weather?",
        "What about tomorrow?",
    }


def test_split_user_input_before_assistant_output_stays_in_one_audio_turn(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    state = _state(tracer_provider)

    state.on_speech_started("item-1")
    state.on_user_transcript_completed("item-1", "First part.")
    state.on_speech_started("item-2")
    state.on_user_transcript_completed("item-2", "Second part.")
    state.on_response_created("response-1", None)
    state.on_asst_transcript_done("response-1", "Combined answer.")
    state.on_response_done("response-1", None, None)
    state.on_session_close()

    spans = _spans_by_kind(in_memory_span_exporter)

    assert len(spans[_AUDIO_KIND]) == 1
    assert len(spans[_USER_KIND]) == 2
    assert len(spans[OpenInferenceSpanKindValues.LLM.value]) == 1
    assert (
        spans[_AUDIO_KIND][0].attributes[SpanAttributes.INPUT_VALUE] == "First part.\nSecond part."
    )


def test_text_input_creates_user_child_span(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    state = _state(tracer_provider)

    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "conversation.item.created",
                "item": {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Summarize this."}],
                },
            }
        ),
    )
    state.on_response_created("response-1", None)
    state.on_asst_transcript_done("response-1", "Summary.")
    state.on_response_done("response-1", None, None)
    state.on_session_close()

    spans = _spans_by_kind(in_memory_span_exporter)
    user_spans = spans[_USER_KIND]

    assert len(spans[_AUDIO_KIND]) == 1
    assert len(user_spans) == 1
    assert user_spans[0].attributes[SpanAttributes.INPUT_VALUE] == "Summarize this."


def test_realtime_spans_inherit_session_context_from_session_creation(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with using_session("realtime-session-1"):
        state = _state(tracer_provider)

    state.on_speech_started("item-1")
    state.on_user_transcript_completed("item-1", "Hello.")
    state.on_response_created("response-1", None)
    state.on_asst_transcript_done("response-1", "Hi.")
    state.on_response_done("response-1", None, None)
    state.on_session_close()

    for span in in_memory_span_exporter.get_finished_spans():
        assert span.attributes is not None
        assert span.attributes[SpanAttributes.SESSION_ID] == "realtime-session-1"


def test_realtime_spans_use_provider_session_id_when_context_has_none(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    state = _state(tracer_provider)
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "session.created",
                "session": {"id": "sess_provider_1"},
            }
        ),
    )

    state.on_speech_started("item-1")
    state.on_user_transcript_completed("item-1", "Hello.")
    state.on_response_created("response-1", None)
    state.on_asst_transcript_done("response-1", "Hi.")
    state.on_response_done("response-1", None, None)
    state.on_session_close()

    for span in in_memory_span_exporter.get_finished_spans():
        assert span.attributes is not None
        assert span.attributes[SpanAttributes.SESSION_ID] == "sess_provider_1"


def test_realtime_context_session_id_takes_precedence_over_provider_session_id(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with using_session("user-session-1"):
        state = _state(tracer_provider)
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "session.created",
                "session": {"id": "sess_provider_1"},
            }
        ),
    )

    state.on_speech_started("item-1")
    state.on_user_transcript_completed("item-1", "Hello.")
    state.on_response_created("response-1", None)
    state.on_asst_transcript_done("response-1", "Hi.")
    state.on_response_done("response-1", None, None)
    state.on_session_close()

    for span in in_memory_span_exporter.get_finished_spans():
        assert span.attributes is not None
        assert span.attributes[SpanAttributes.SESSION_ID] == "user-session-1"


def test_session_config_captured_on_turn_invocation_parameters(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """session.created config (instructions, voice, etc.) must land on the AUDIO turn span
    as llm.invocation_parameters JSON. Model goes to top-level llm.model_name; tools and
    tool_choice are excluded entirely."""
    import json

    state = _state(tracer_provider)
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "session.created",
                "session": {
                    "id": "sess_abc",
                    "model": "gpt-realtime",
                    "instructions": "You are a helpful voice assistant.",
                    "voice": "alloy",
                    "modalities": ["audio", "text"],
                    "temperature": 0.8,
                    "turn_detection": {"type": "server_vad", "threshold": 0.5},
                    "tools": [{"type": "function", "name": "get_weather"}],
                    "tool_choice": "auto",
                    "client_secret": "SHOULD_NOT_LEAK",
                },
            }
        ),
    )
    state.on_speech_started("item-1")
    state.on_user_transcript_completed("item-1", "hi")
    state.on_response_created("response-1", "gpt-realtime")
    state.on_asst_transcript_done("response-1", "hello")
    state.on_response_done("response-1", None, "gpt-realtime")
    state.on_session_close()

    audio_spans = _spans_by_kind(in_memory_span_exporter)[_AUDIO_KIND]
    assert len(audio_spans) == 1
    attrs = audio_spans[0].attributes
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "gpt-realtime"
    params = json.loads(attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS])
    assert params["instructions"] == "You are a helpful voice assistant."
    assert params["voice"] == "alloy"
    assert params["modalities"] == ["audio", "text"]
    assert params["temperature"] == 0.8
    assert params["turn_detection"] == {"type": "server_vad", "threshold": 0.5}
    assert "model" not in params
    assert "tools" not in params
    assert "tool_choice" not in params
    assert "client_secret" not in params
    assert "id" not in params


def test_session_updated_overrides_earlier_config(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """session.updated must merge into the captured config; later turns see new values."""
    import json

    state = _state(tracer_provider)
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "session.created",
                "session": {"id": "sess_abc", "instructions": "First.", "voice": "alloy"},
            }
        ),
    )
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "session.updated",
                "session": {"instructions": "Second."},
            }
        ),
    )
    state.on_speech_started("item-1")
    state.on_user_transcript_completed("item-1", "hi")
    state.on_response_created("response-1", None)
    state.on_asst_transcript_done("response-1", "hello")
    state.on_response_done("response-1", None, None)
    state.on_session_close()

    audio_spans = _spans_by_kind(in_memory_span_exporter)[_AUDIO_KIND]
    params = json.loads(audio_spans[0].attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS])
    assert params["instructions"] == "Second."
    assert params["voice"] == "alloy"


def test_time_to_first_token_ms_measures_from_end_of_user_query_to_first_audio(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: Any,
) -> None:
    """time_to_first_token_ms = (first audio delta time) - (user audio committed time).

    Captures perceived latency: silence from when the user stopped talking to when the
    assistant's first audio chunk arrives. NOT 'response.created → first delta' (model
    decode only), which would skip server VAD silence-detection time.
    """
    fake_now_ns = [0]

    def fake_monotonic_ns() -> int:
        return fake_now_ns[0]

    monkeypatch.setattr(
        "openinference.instrumentation.openai_agents._realtime.time.monotonic_ns",
        fake_monotonic_ns,
    )

    state = _state(tracer_provider)

    fake_now_ns[0] = 0
    state.on_speech_started("item-1")

    # User finishes talking at t=1000ms; server VAD commits the buffer.
    fake_now_ns[0] = 1_000_000_000  # 1.0 s
    state.on_user_audio_committed("item-1")
    state.on_user_transcript_completed("item-1", "hi")

    # Server announces response at t=1050ms (50 ms VAD gap).
    fake_now_ns[0] = 1_050_000_000  # 1.05 s
    state.on_response_created("response-1", "gpt-realtime")

    # First audio chunk arrives at t=1500ms.
    fake_now_ns[0] = 1_500_000_000  # 1.5 s
    state.on_audio_delta("response-1", b"\x00\x00")

    fake_now_ns[0] = 1_600_000_000
    state.on_asst_transcript_done("response-1", "hello")
    state.on_response_done("response-1", None, "gpt-realtime")
    state.on_session_close()

    llm_spans = _spans_by_kind(in_memory_span_exporter)[OpenInferenceSpanKindValues.LLM.value]
    assert len(llm_spans) == 1
    ttft_ms = llm_spans[0].attributes["time_to_first_token_ms"]
    # Expect 500 ms (1500 - 1000), not 450 ms (1500 - 1050).
    assert ttft_ms == 500


def test_agent_end_event_does_not_finalize_freshly_opened_barge_in_turn(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: Any,
) -> None:
    """Reproduces the user-reported 3-span barge-in trace.

    When a user interrupts mid-playback, the assistant's response already
    completed server-side (server.response.done fired), so turn 1 finalizes as
    COMPLETE on the new speech_started. The barge-in opens turn 2.
    RealtimeAgentEndEvent then fires (the prior agent's logical run ended due
    to interruption). It MUST NOT finalize turn 2 — turn 2 belongs to the new
    user input, not the agent that just ended.

    Bug: _dispatch_event treats RealtimeAgentEndEvent like a session close,
    which prematurely closes turn 2 as an empty USER-only span.
    """

    class _FakeAgentEndEvent:
        pass

    monkeypatch.setattr(_realtime, "_RealtimeAgentEndEvent", _FakeAgentEndEvent)

    state = _state(tracer_provider)

    # Turn 1 — WoW Q&A completes server-side normally.
    state.on_speech_started("item-1")
    state.on_user_audio_committed("item-1")
    state.on_user_transcript_completed("item-1", "Tell me about WoW.")
    state.on_response_created("response-1", "gpt-realtime")
    state.on_asst_transcript_done("response-1", "World of Warcraft is …")
    state.on_response_done("response-1", None, "gpt-realtime")

    # User cuts in mid-playback. Server VAD fires speech_started → turn 1
    # finalizes (COMPLETE — its responses are closed), turn 2 opens.
    state.on_speech_started("item-2")

    # SDK fires RealtimeAgentEndEvent because the prior WoW agent run "ended"
    # via interruption. This should NOT finalize turn 2.
    _dispatch_event(state, _FakeAgentEndEvent())

    # The real barge-in continues. User finishes, transcript completes,
    # assistant responds.
    state.on_user_audio_committed("item-2")
    state.on_user_transcript_completed("item-2", "No, actually, I meant Starcraft.")
    state.on_response_created("response-2", "gpt-realtime")
    state.on_asst_transcript_done("response-2", "StarCraft is …")
    state.on_response_done("response-2", None, "gpt-realtime")
    state.on_session_close()

    spans = _spans_by_kind(in_memory_span_exporter)
    audio_spans = spans[_AUDIO_KIND]
    # Two real turns, not three. No phantom empty turn between them.
    assert len(audio_spans) == 2
    inputs = [s.attributes.get(SpanAttributes.INPUT_VALUE) for s in audio_spans]
    outputs = [s.attributes.get(SpanAttributes.OUTPUT_VALUE) for s in audio_spans]
    assert "Tell me about WoW." in inputs
    assert "No, actually, I meant Starcraft." in inputs
    assert "World of Warcraft is …" in outputs
    assert "StarCraft is …" in outputs


def test_agent_end_event_is_noop_on_open_turn(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: Any,
) -> None:
    """RealtimeAgentEndEvent must not finalize the current turn on its own.

    Tightly scoped variant: open a turn with a USER child, fire AgentEndEvent,
    and assert no AUDIO/USER spans have been exported yet (turn still open).
    """

    class _FakeAgentEndEvent:
        pass

    monkeypatch.setattr(_realtime, "_RealtimeAgentEndEvent", _FakeAgentEndEvent)

    state = _state(tracer_provider)
    state.on_speech_started("item-1")
    state.on_send_audio(b"\x00" * 1024)

    _dispatch_event(state, _FakeAgentEndEvent())

    spans = _spans_by_kind(in_memory_span_exporter)
    assert _AUDIO_KIND not in spans
    assert _USER_KIND not in spans


def test_interrupted_response_captures_partial_transcript_from_deltas(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """When a response is interrupted, response.output_audio_transcript.done never
    fires — only .delta events arrived. We must accumulate the deltas so the partial
    transcript is captured on both the LLM child span and the parent AUDIO span.
    """
    state = _state(tracer_provider)

    state.on_speech_started("item-1")
    state.on_user_audio_committed("item-1")
    state.on_user_transcript_completed("item-1", "Tell me about WoW.")

    _dispatch_raw(
        state,
        _raw_event(
            {"type": "response.created", "response": {"id": "response-1", "model": "gpt-realtime"}}
        ),
    )
    # Partial transcript streams in via deltas (no .done because user interrupts).
    for delta in ("World of Warcraft ", "is a huge online ", "fantasy game"):
        _dispatch_raw(
            state,
            _raw_event(
                {
                    "type": "response.output_audio_transcript.delta",
                    "response_id": "response-1",
                    "delta": delta,
                }
            ),
        )
    # Server cancels response. response.done arrives with no .done for the transcript.
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {"id": "response-1", "model": "gpt-realtime", "status": "cancelled"},
            }
        ),
    )
    state.on_session_close()

    spans = _spans_by_kind(in_memory_span_exporter)
    llm_spans = spans[OpenInferenceSpanKindValues.LLM.value]
    audio_spans = spans[_AUDIO_KIND]
    expected = "World of Warcraft is a huge online fantasy game"
    assert llm_spans[0].attributes["output.audio.transcript"] == expected
    assert audio_spans[0].attributes[SpanAttributes.OUTPUT_VALUE] == expected


def test_transcript_done_overrides_accumulated_deltas(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """When .done fires normally, its full transcript wins over accumulated deltas.

    Deltas may not perfectly reconstruct the final transcript (e.g., server-side
    normalization), so the authoritative .done text should take precedence.
    """
    state = _state(tracer_provider)
    state.on_speech_started("item-1")
    state.on_user_audio_committed("item-1")
    state.on_user_transcript_completed("item-1", "Hi.")
    _dispatch_raw(
        state,
        _raw_event(
            {"type": "response.created", "response": {"id": "response-1", "model": "gpt-realtime"}}
        ),
    )
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.output_audio_transcript.delta",
                "response_id": "response-1",
                "delta": "hello",
            }
        ),
    )
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.output_audio_transcript.done",
                "response_id": "response-1",
                "transcript": "Hello there!",
            }
        ),
    )
    _dispatch_raw(
        state,
        _raw_event(
            {"type": "response.done", "response": {"id": "response-1", "model": "gpt-realtime"}}
        ),
    )
    state.on_session_close()

    llm_spans = _spans_by_kind(in_memory_span_exporter)[OpenInferenceSpanKindValues.LLM.value]
    assert llm_spans[0].attributes["output.audio.transcript"] == "Hello there!"


def test_spurious_audio_interrupted_after_completed_response_is_ignored(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """SDK fires RealtimeAudioInterrupted on every new user turn (its tracker keeps
    referencing the last assistant audio item even after response.done). We must
    only mark a turn INTERRUPTED when there's an actively in-flight assistant
    response — otherwise normal follow-up questions get tagged INTERRUPTED.
    """
    state = _state(tracer_provider)

    # Turn 1: normal Q&A with a tool call (multi-response) that fully completes.
    state.on_speech_started("item-1")
    state.on_user_audio_committed("item-1")
    state.on_user_transcript_completed("item-1", "What's the weather in Tokyo?")
    state.on_response_created("response-1a", "gpt-realtime")
    state.on_asst_transcript_done("response-1a", "Let me check the weather…")
    state.on_response_done("response-1a", None, "gpt-realtime")
    state.on_response_created("response-1b", "gpt-realtime")
    state.on_asst_transcript_done("response-1b", "It's 72 and sunny in Tokyo.")
    state.on_response_done("response-1b", None, "gpt-realtime")

    # User asks a follow-up. speech_started finalizes turn 1 (all responses closed)
    # and opens turn 2.
    state.on_speech_started("item-2")
    # Spurious SDK event — fires because the tracker still has the previous
    # assistant audio item, NOT because anything was actually interrupted.
    state.on_audio_interrupted()
    state.on_user_audio_committed("item-2")
    state.on_user_transcript_completed("item-2", "What time is it in Tokyo?")
    state.on_response_created("response-2", "gpt-realtime")
    state.on_asst_transcript_done("response-2", "It's 2:37 PM in Tokyo.")
    state.on_response_done("response-2", None, "gpt-realtime")
    state.on_session_close()

    audio_spans = _spans_by_kind(in_memory_span_exporter)[_AUDIO_KIND]
    assert len(audio_spans) == 2
    end_reasons = [s.attributes["end_reason"] for s in audio_spans]
    assert end_reasons == ["complete", "complete"]


def test_audio_interrupted_during_in_flight_response_marks_turn(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """The legitimate case: when an assistant response is still in flight and the
    SDK fires RealtimeAudioInterrupted, mark the turn INTERRUPTED.
    """
    state = _state(tracer_provider)
    state.on_speech_started("item-1")
    state.on_user_audio_committed("item-1")
    state.on_user_transcript_completed("item-1", "Tell me about WoW.")
    state.on_response_created("response-1", "gpt-realtime")
    # Partial transcript via delta; response.done has not yet arrived → in-flight.
    state.on_asst_transcript_delta("response-1", "World of Warcraft is …")
    state.on_audio_interrupted()
    # Server cancels the in-flight response.
    state.on_response_done("response-1", None, "gpt-realtime")
    state.on_session_close()

    audio_spans = _spans_by_kind(in_memory_span_exporter)[_AUDIO_KIND]
    assert audio_spans[0].attributes["end_reason"] == "interrupted"


def _function_call_output() -> dict[str, Any]:
    """A response.done output item representing a tool call (no audio)."""
    return {
        "type": "function_call",
        "name": "get_current_time",
        "call_id": "call-1",
        "arguments": '{"timezone": "Asia/Tokyo"}',
    }


def test_tool_followup_response_routes_to_originating_turn_when_no_barge_in(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Baseline: tool round-trip with no interruption produces one AUDIO turn
    with two LLM children (pre-tool + post-tool) and a TOOL child between."""
    state = _state(tracer_provider)
    state.on_speech_started("item-1")
    state.on_user_audio_committed("item-1")
    state.on_user_transcript_completed("item-1", "What's the time in Tokyo?")
    state.on_response_created("response-1a", "gpt-realtime")
    state.on_asst_transcript_done("response-1a", "Let me check the time.")
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {
                    "id": "response-1a",
                    "model": "gpt-realtime",
                    "output": [_function_call_output()],
                },
            }
        ),
    )
    state.on_response_created("response-1b", "gpt-realtime")
    state.on_asst_transcript_done("response-1b", "It's 2:47 PM in Tokyo.")
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {"id": "response-1b", "model": "gpt-realtime", "output": []},
            }
        ),
    )
    state.on_session_close()

    spans = _spans_by_kind(in_memory_span_exporter)
    audio_spans = spans[_AUDIO_KIND]
    llm_spans = spans[OpenInferenceSpanKindValues.LLM.value]
    assert len(audio_spans) == 1
    assert len(llm_spans) == 2
    audio_span_id = audio_spans[0].context.span_id
    assert all(s.parent.span_id == audio_span_id for s in llm_spans)
    assert audio_spans[0].attributes["end_reason"] == "complete"
    assert audio_spans[0].attributes[SpanAttributes.OUTPUT_VALUE] == (
        "Let me check the time.\nIt's 2:47 PM in Tokyo."
    )


def test_barge_in_during_tool_gap_keeps_followup_on_originating_turn(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Case B: user barges in DURING the gap between response_1a (function_call)
    and response_1b. The follow-up MUST still land on turn 1, and turn 1's
    end_reason MUST be `interrupted`.
    """
    state = _state(tracer_provider)
    # Turn 1 — user asks Tokyo time, model calls tool, tool resolves server-side.
    state.on_speech_started("item-1")
    state.on_user_audio_committed("item-1")
    state.on_user_transcript_completed("item-1", "What's the time in Tokyo?")
    state.on_response_created("response-1a", "gpt-realtime")
    state.on_asst_transcript_done("response-1a", "Let me check the time.")
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {
                    "id": "response-1a",
                    "model": "gpt-realtime",
                    "output": [_function_call_output()],
                },
            }
        ),
    )

    # User barges in DURING the gap (response_1b hasn't arrived yet).
    state.on_speech_started("item-2")

    # Follow-up arrives AFTER barge-in opened turn 2.
    state.on_response_created("response-1b", "gpt-realtime")
    state.on_asst_transcript_done("response-1b", "It's 2:47 PM in Tokyo.")
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {"id": "response-1b", "model": "gpt-realtime", "output": []},
            }
        ),
    )

    # Turn 2 continues normally.
    state.on_user_audio_committed("item-2")
    state.on_user_transcript_completed("item-2", "What's the weather in Tokyo?")
    state.on_response_created("response-2", "gpt-realtime")
    state.on_asst_transcript_done("response-2", "It's 72 and sunny.")
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {"id": "response-2", "model": "gpt-realtime", "output": []},
            }
        ),
    )
    state.on_session_close()

    spans = _spans_by_kind(in_memory_span_exporter)
    audio_spans = sorted(spans[_AUDIO_KIND], key=lambda s: s.start_time)
    assert len(audio_spans) == 2

    turn1, turn2 = audio_spans
    assert turn1.attributes[SpanAttributes.INPUT_VALUE] == "What's the time in Tokyo?"
    assert turn1.attributes[SpanAttributes.OUTPUT_VALUE] == (
        "Let me check the time.\nIt's 2:47 PM in Tokyo."
    )
    assert turn1.attributes["end_reason"] == "interrupted"

    assert turn2.attributes[SpanAttributes.INPUT_VALUE] == "What's the weather in Tokyo?"
    assert turn2.attributes[SpanAttributes.OUTPUT_VALUE] == "It's 72 and sunny."
    assert turn2.attributes["end_reason"] == "complete"

    # Two LLM children in turn 1 (response_1a + response_1b), one in turn 2.
    llm_spans = spans[OpenInferenceSpanKindValues.LLM.value]
    turn1_id = turn1.context.span_id
    turn2_id = turn2.context.span_id
    assert sum(1 for s in llm_spans if s.parent.span_id == turn1_id) == 2
    assert sum(1 for s in llm_spans if s.parent.span_id == turn2_id) == 1


def test_barge_in_during_followup_response_keeps_partial_followup_on_originating_turn(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Case C: user barges in WHILE response_1b is in flight. response_1b's
    partial transcript (via deltas) must land on turn 1, marked interrupted.
    """
    state = _state(tracer_provider)
    state.on_speech_started("item-1")
    state.on_user_audio_committed("item-1")
    state.on_user_transcript_completed("item-1", "What's the time in Tokyo?")
    state.on_response_created("response-1a", "gpt-realtime")
    state.on_asst_transcript_done("response-1a", "Let me check the time.")
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {
                    "id": "response-1a",
                    "model": "gpt-realtime",
                    "output": [_function_call_output()],
                },
            }
        ),
    )
    # Follow-up starts.
    state.on_response_created("response-1b", "gpt-realtime")
    # Partial transcript via deltas — never reaches .done before barge-in.
    state.on_asst_transcript_delta("response-1b", "It's 2:47 PM ")

    # User barges in mid-followup.
    state.on_speech_started("item-2")

    # Server cancels response-1b.
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {
                    "id": "response-1b",
                    "model": "gpt-realtime",
                    "status": "cancelled",
                    "output": [],
                },
            }
        ),
    )

    state.on_user_audio_committed("item-2")
    state.on_user_transcript_completed("item-2", "What's the weather?")
    state.on_response_created("response-2", "gpt-realtime")
    state.on_asst_transcript_done("response-2", "Sunny.")
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {"id": "response-2", "model": "gpt-realtime", "output": []},
            }
        ),
    )
    state.on_session_close()

    spans = _spans_by_kind(in_memory_span_exporter)
    audio_spans = sorted(spans[_AUDIO_KIND], key=lambda s: s.start_time)
    assert len(audio_spans) == 2
    turn1 = audio_spans[0]
    assert turn1.attributes[SpanAttributes.OUTPUT_VALUE] == (
        "Let me check the time.\nIt's 2:47 PM "
    )
    assert turn1.attributes["end_reason"] == "interrupted"


def test_barge_in_before_function_call_finalizes_turn_immediately(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Case A: barge-in mid-response_1a (before any function_call output is
    declared). Standard in-flight interruption — turn 1 finalizes immediately,
    no awaiting-followup state, no orphan routing.
    """
    state = _state(tracer_provider)
    state.on_speech_started("item-1")
    state.on_user_audio_committed("item-1")
    state.on_user_transcript_completed("item-1", "Tell me a story.")
    state.on_response_created("response-1", "gpt-realtime")
    state.on_asst_transcript_delta("response-1", "Once upon a time ")
    # User barges in mid-response, no function_call yet.
    state.on_speech_started("item-2")
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {
                    "id": "response-1",
                    "model": "gpt-realtime",
                    "status": "cancelled",
                    "output": [],
                },
            }
        ),
    )
    state.on_user_audio_committed("item-2")
    state.on_user_transcript_completed("item-2", "Never mind.")
    state.on_response_created("response-2", "gpt-realtime")
    state.on_asst_transcript_done("response-2", "OK.")
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {"id": "response-2", "model": "gpt-realtime", "output": []},
            }
        ),
    )
    state.on_session_close()

    spans = _spans_by_kind(in_memory_span_exporter)
    audio_spans = sorted(spans[_AUDIO_KIND], key=lambda s: s.start_time)
    assert len(audio_spans) == 2
    assert audio_spans[0].attributes["end_reason"] == "interrupted"
    assert audio_spans[0].attributes[SpanAttributes.OUTPUT_VALUE] == "Once upon a time "
    assert audio_spans[1].attributes["end_reason"] == "complete"


def test_tool_call_creates_child_tool_span_under_llm(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """When the model calls a tool, a TOOL span must appear as a child of the
    LLM span that triggered it, with tool name, parameters (JSON args), and
    output (JSON result).
    """
    state = _state(tracer_provider)
    state.on_speech_started("item-1")
    state.on_user_audio_committed("item-1")
    state.on_user_transcript_completed("item-1", "What's the time in Tokyo?")
    state.on_response_created("response-1a", "gpt-realtime")
    state.on_asst_transcript_done("response-1a", "Let me check the time.")

    # Server announces the function_call output item.
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.output_item.added",
                "response_id": "response-1a",
                "item": {
                    "type": "function_call",
                    "id": "call-1",
                    "name": "get_current_time",
                },
            }
        ),
    )
    # Arguments finalize.
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.function_call_arguments.done",
                "response_id": "response-1a",
                "call_id": "call-1",
                "arguments": '{"timezone": "Asia/Tokyo"}',
            }
        ),
    )
    # response-1a completes with function_call output.
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {
                    "id": "response-1a",
                    "model": "gpt-realtime",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "call-1",
                            "name": "get_current_time",
                        }
                    ],
                },
            }
        ),
    )
    # Tool result submitted by the agent SDK as a new conversation item.
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "conversation.item.created",
                "item": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": '{"timezone": "Asia/Tokyo", "time": "14:47"}',
                },
            }
        ),
    )
    # Follow-up response.
    state.on_response_created("response-1b", "gpt-realtime")
    state.on_asst_transcript_done("response-1b", "It's 2:47 PM in Tokyo.")
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {"id": "response-1b", "model": "gpt-realtime", "output": []},
            }
        ),
    )
    state.on_session_close()

    spans = _spans_by_kind(in_memory_span_exporter)
    tool_spans = spans.get(OpenInferenceSpanKindValues.TOOL.value, [])
    assert len(tool_spans) == 1
    tool_span = tool_spans[0]

    # Tool span attributes.
    assert tool_span.name == "get_current_time"
    assert tool_span.attributes[SpanAttributes.TOOL_NAME] == "get_current_time"
    assert tool_span.attributes[SpanAttributes.INPUT_VALUE] == '{"timezone": "Asia/Tokyo"}'
    assert tool_span.attributes[SpanAttributes.OUTPUT_VALUE] == (
        '{"timezone": "Asia/Tokyo", "time": "14:47"}'
    )

    # Parent is the LLM span for response-1a (the one that triggered the tool).
    llm_spans = spans[OpenInferenceSpanKindValues.LLM.value]
    pre_tool_llm = next(
        s
        for s in llm_spans
        if s.attributes.get("output.audio.transcript") == "Let me check the time."
    )
    assert tool_span.parent.span_id == pre_tool_llm.context.span_id


def test_tool_call_output_via_conversation_item_added(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """OpenAI Realtime GA API emits `conversation.item.added` (not `.created`)
    for function_call_output items. We must handle both event names.
    """
    state = _state(tracer_provider)
    state.on_speech_started("item-1")
    state.on_user_audio_committed("item-1")
    state.on_user_transcript_completed("item-1", "Time in Tokyo?")
    state.on_response_created("response-1a", "gpt-realtime")
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.output_item.added",
                "response_id": "response-1a",
                "item": {
                    "type": "function_call",
                    "id": "call-1",
                    "name": "get_current_time",
                },
            }
        ),
    )
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.function_call_arguments.done",
                "response_id": "response-1a",
                "call_id": "call-1",
                "arguments": '{"timezone": "Asia/Tokyo"}',
            }
        ),
    )
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {
                    "id": "response-1a",
                    "model": "gpt-realtime",
                    "output": [
                        {"type": "function_call", "id": "call-1", "name": "get_current_time"}
                    ],
                },
            }
        ),
    )
    # GA API emits `.added` not `.created`.
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "conversation.item.added",
                "item": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": '{"time": "14:47"}',
                },
            }
        ),
    )
    state.on_response_created("response-1b", "gpt-realtime")
    state.on_asst_transcript_done("response-1b", "It's 2:47 PM.")
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {"id": "response-1b", "model": "gpt-realtime", "output": []},
            }
        ),
    )
    state.on_session_close()

    tool_spans = _spans_by_kind(in_memory_span_exporter).get(
        OpenInferenceSpanKindValues.TOOL.value, []
    )
    assert len(tool_spans) == 1
    assert tool_spans[0].attributes[SpanAttributes.OUTPUT_VALUE] == '{"time": "14:47"}'


def test_tool_call_with_distinct_item_id_and_call_id(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Real-world OpenAI Realtime events use distinct item.id and item.call_id.
    The function_call_output linkage uses call_id. Our tool span lookup must
    key off call_id, not item.id.
    """
    state = _state(tracer_provider)
    state.on_speech_started("user-item")
    state.on_user_audio_committed("user-item")
    state.on_user_transcript_completed("user-item", "Weather in Tokyo?")
    state.on_response_created("response-1a", "gpt-realtime")
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.output_item.added",
                "response_id": "response-1a",
                "item": {
                    "id": "item_fc_001",
                    "call_id": "call_3VtAn",  # DISTINCT from item.id
                    "type": "function_call",
                    "name": "get_weather",
                },
            }
        ),
    )
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.function_call_arguments.done",
                "response_id": "response-1a",
                "item_id": "item_fc_001",
                "call_id": "call_3VtAn",
                "arguments": '{"location":"Tokyo","unit":"fahrenheit"}',
            }
        ),
    )
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {
                    "id": "response-1a",
                    "model": "gpt-realtime",
                    "output": [
                        {
                            "id": "item_fc_001",
                            "call_id": "call_3VtAn",
                            "type": "function_call",
                            "name": "get_weather",
                        }
                    ],
                },
            }
        ),
    )
    # Server echoes the function_call_output keyed by call_id.
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "conversation.item.added",
                "item": {
                    "id": "item_fco_001",
                    "type": "function_call_output",
                    "call_id": "call_3VtAn",
                    "output": "The weather in Tokyo is 72 °F and sunny.",
                },
            }
        ),
    )
    state.on_response_created("response-1b", "gpt-realtime")
    state.on_asst_transcript_done("response-1b", "It's 72 and sunny.")
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {"id": "response-1b", "model": "gpt-realtime", "output": []},
            }
        ),
    )
    state.on_session_close()

    tool_spans = _spans_by_kind(in_memory_span_exporter).get(
        OpenInferenceSpanKindValues.TOOL.value, []
    )
    assert len(tool_spans) == 1
    tool_span = tool_spans[0]
    assert tool_span.attributes[SpanAttributes.TOOL_NAME] == "get_weather"
    assert tool_span.attributes[SpanAttributes.INPUT_VALUE] == (
        '{"location":"Tokyo","unit":"fahrenheit"}'
    )
    assert tool_span.attributes[SpanAttributes.OUTPUT_VALUE] == (
        "The weather in Tokyo is 72 °F and sunny."
    )


def test_duplicate_text_item_events_produce_one_user_span(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Server emits both `conversation.item.added` AND `.done` for a single
    client-sent text item. Both share the same `item.id`. We must produce
    ONE USER span, not two.
    """
    state = _state(tracer_provider)
    text_item = {
        "id": "item_text_001",
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "talk like a pirate"}],
    }
    _dispatch_raw(state, _raw_event({"type": "conversation.item.added", "item": text_item}))
    _dispatch_raw(state, _raw_event({"type": "conversation.item.done", "item": text_item}))
    state.on_response_created("response-1", "gpt-realtime")
    state.on_asst_transcript_done("response-1", "Arrr.")
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {"id": "response-1", "model": "gpt-realtime", "output": []},
            }
        ),
    )
    state.on_session_close()

    spans = _spans_by_kind(in_memory_span_exporter)
    user_spans = spans[_USER_KIND]
    assert len(user_spans) == 1
    assert user_spans[0].attributes[SpanAttributes.INPUT_VALUE] == "talk like a pirate"


def test_transcript_completed_routes_to_speech_user_by_item_id(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """When text input interleaves with speech, the audio-transcription
    `completed` event must land on the USER created for that speech item_id —
    not on the most-recently-active text USER.
    """
    state = _state(tracer_provider)
    # Speech starts first (creates audio USER tied to item_audio_001).
    state.on_speech_started("item_audio_001")
    # Then text arrives mid-utterance (a separate USER child).
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "conversation.item.added",
                "item": {
                    "id": "item_text_001",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "talk like a pirate"}],
                },
            }
        ),
    )
    # Speech commits and transcript completes (carrying item_audio_001).
    state.on_user_audio_committed("item_audio_001")
    state.on_user_transcript_completed("item_audio_001", "What's the weather in Tokyo?")
    state.on_response_created("response-1", "gpt-realtime")
    state.on_asst_transcript_done("response-1", "Arrr, sunny.")
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {"id": "response-1", "model": "gpt-realtime", "output": []},
            }
        ),
    )
    state.on_session_close()

    user_spans = _spans_by_kind(in_memory_span_exporter)[_USER_KIND]
    # Find the USER carrying the audio transcript.
    audio_users = [s for s in user_spans if s.attributes.get("input.audio.transcript") is not None]
    text_users = [
        s
        for s in user_spans
        if s.attributes.get(SpanAttributes.INPUT_VALUE) == "talk like a pirate"
    ]
    assert len(audio_users) == 1
    assert len(text_users) == 1
    # The audio USER must own the transcript; the text USER must NOT.
    assert audio_users[0] is not text_users[0]
    assert audio_users[0].attributes["input.audio.transcript"] == ("What's the weather in Tokyo?")
    assert text_users[0].attributes.get("input.audio.transcript") is None


def test_text_only_user_span_does_not_carry_audio_attributes(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Text-input USER spans must NOT carry input.audio.url or .mime_type.
    Any mic audio that flows in while a text USER is `active` should go to
    the prespeech buffer instead, not onto the text USER's audio buffer.
    """
    state = _state(tracer_provider)

    # Text input creates a text USER.
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "conversation.item.added",
                "item": {
                    "id": "item_text_001",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hi there"}],
                },
            }
        ),
    )
    # Mic audio happens to be flowing while the text USER is active.
    state.on_send_audio(b"\x00\x01" * 512)

    state.on_response_created("response-1", "gpt-realtime")
    state.on_asst_transcript_done("response-1", "Hello!")
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "response.done",
                "response": {"id": "response-1", "model": "gpt-realtime", "output": []},
            }
        ),
    )
    state.on_session_close()

    user_spans = _spans_by_kind(in_memory_span_exporter)[_USER_KIND]
    assert len(user_spans) == 1
    attrs = user_spans[0].attributes
    assert attrs[SpanAttributes.INPUT_VALUE] == "Hi there"
    # No audio attributes whatsoever on a text-only USER.
    assert "input.audio.url" not in attrs
    assert "input.audio.mime_type" not in attrs
    assert "input.audio.transcript" not in attrs


def test_close_wrapper_finalizes_open_turn(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """RealtimeSession.close() must end the parent AUDIO and USER spans.

    Without this hook, a clean session shutdown (Ctrl-C, async-with exit) leaves
    the turn open and the spans are never exported.
    """
    state = _state(tracer_provider)

    state.on_speech_started("item-1")
    state.on_user_transcript_completed("item-1", "Hello.")
    state.on_response_created("response-1", "gpt-realtime")
    state.on_asst_transcript_done("response-1", "Hi.")
    state.on_response_done("response-1", None, "gpt-realtime")

    # Before close: only LLM child is ended; AUDIO/USER stay open.
    spans = _spans_by_kind(in_memory_span_exporter)
    assert _AUDIO_KIND not in spans
    assert _USER_KIND not in spans
    assert len(spans[OpenInferenceSpanKindValues.LLM.value]) == 1

    # Simulate the wrapt-wrapped close path.
    class _FakeSession:
        pass

    session: Any = _FakeSession()
    _session_states[session] = state

    async def _original_close(*args: Any, **kwargs: Any) -> None:
        return None

    wrapper = make_close_wrapper()
    asyncio.run(wrapper(_original_close, session, (), {}))

    spans = _spans_by_kind(in_memory_span_exporter)
    assert len(spans[_AUDIO_KIND]) == 1
    assert len(spans[_USER_KIND]) == 1


# ----------------------------------------------------------------------
# Audio redaction + truncation
# ----------------------------------------------------------------------


def test_truncate_audio_data_uri_preserves_prefix_and_truncates_payload() -> None:
    uri = "data:audio/wav;base64," + ("A" * 500)
    out = truncate_audio_data_uri(uri, 64)
    prefix, payload = out.split(";base64,", 1)
    assert prefix == "data:audio/wav"
    assert payload == "A" * 64


def test_truncate_audio_data_uri_no_op_when_below_limit() -> None:
    uri = "data:audio/wav;base64,AAAA"
    assert truncate_audio_data_uri(uri, 64) == uri


def test_pcm16_to_wav_data_uri_produces_data_uri() -> None:
    uri = pcm16_to_wav_data_uri(b"\x00\x01" * 32)
    assert uri.startswith("data:audio/wav;base64,")
    assert len(uri) > len("data:audio/wav;base64,")


def test_audio_emitted_by_default(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    state = _state(tracer_provider)
    _drive_audio_turn(state)

    user_attrs = _spans_by_kind(in_memory_span_exporter)[_USER_KIND][0].attributes
    llm_attrs = _spans_by_kind(in_memory_span_exporter)[OpenInferenceSpanKindValues.LLM.value][
        0
    ].attributes
    assert user_attrs["input.audio.url"].startswith("data:audio/wav;base64,")
    assert user_attrs["input.audio.mime_type"] == "audio/wav"
    assert llm_attrs["output.audio.url"].startswith("data:audio/wav;base64,")
    assert llm_attrs["output.audio.mime_type"] == "audio/wav"


def test_base64_audio_max_length_env_truncates_payload(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH", "32")
    state = _state(tracer_provider)
    _drive_audio_turn(state)

    user_attrs = _spans_by_kind(in_memory_span_exporter)[_USER_KIND][0].attributes
    llm_attrs = _spans_by_kind(in_memory_span_exporter)[OpenInferenceSpanKindValues.LLM.value][
        0
    ].attributes
    for attrs, key in (
        (user_attrs, "input.audio.url"),
        (llm_attrs, "output.audio.url"),
    ):
        uri = attrs[key]
        prefix, payload = uri.split(";base64,", 1)
        assert prefix == "data:audio/wav"
        assert len(payload) == 32


def test_hide_input_audio_env_drops_input_audio_attrs(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("OPENINFERENCE_HIDE_INPUT_AUDIO", "true")
    state = _state(tracer_provider)
    _drive_audio_turn(state)

    user_attrs = _spans_by_kind(in_memory_span_exporter)[_USER_KIND][0].attributes
    llm_attrs = _spans_by_kind(in_memory_span_exporter)[OpenInferenceSpanKindValues.LLM.value][
        0
    ].attributes
    assert "input.audio.url" not in user_attrs
    assert "input.audio.mime_type" not in user_attrs
    assert "input.audio.transcript" not in user_attrs
    # Output audio is unaffected.
    assert llm_attrs["output.audio.url"].startswith("data:audio/wav;base64,")
    assert llm_attrs["output.audio.transcript"] == "Hello!"


def test_hide_output_audio_env_drops_output_audio_attrs(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("OPENINFERENCE_HIDE_OUTPUT_AUDIO", "true")
    state = _state(tracer_provider)
    _drive_audio_turn(state)

    user_attrs = _spans_by_kind(in_memory_span_exporter)[_USER_KIND][0].attributes
    llm_attrs = _spans_by_kind(in_memory_span_exporter)[OpenInferenceSpanKindValues.LLM.value][
        0
    ].attributes
    assert "output.audio.url" not in llm_attrs
    assert "output.audio.mime_type" not in llm_attrs
    assert "output.audio.transcript" not in llm_attrs
    # Input audio is unaffected.
    assert user_attrs["input.audio.url"].startswith("data:audio/wav;base64,")
    assert user_attrs["input.audio.transcript"] == "Hi."


def test_hide_inputs_drops_input_audio_attrs(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """TraceConfig(hide_inputs=True) implies hide_input_audio."""
    state = _state(tracer_provider, config=TraceConfig(hide_inputs=True))
    _drive_audio_turn(state)

    user_attrs = _spans_by_kind(in_memory_span_exporter)[_USER_KIND][0].attributes
    assert "input.audio.url" not in user_attrs
    assert "input.audio.mime_type" not in user_attrs
    assert "input.audio.transcript" not in user_attrs


def test_hide_outputs_drops_output_audio_attrs(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """TraceConfig(hide_outputs=True) implies hide_output_audio."""
    state = _state(tracer_provider, config=TraceConfig(hide_outputs=True))
    _drive_audio_turn(state)

    llm_attrs = _spans_by_kind(in_memory_span_exporter)[OpenInferenceSpanKindValues.LLM.value][
        0
    ].attributes
    assert "output.audio.url" not in llm_attrs
    assert "output.audio.mime_type" not in llm_attrs
    assert "output.audio.transcript" not in llm_attrs


def test_hide_input_audio_env_off_value_is_no_op(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: Any,
) -> None:
    """Env var with falsy value must NOT trigger redaction."""
    monkeypatch.setenv("OPENINFERENCE_HIDE_INPUT_AUDIO", "false")
    state = _state(tracer_provider)
    _drive_audio_turn(state)

    user_attrs = _spans_by_kind(in_memory_span_exporter)[_USER_KIND][0].attributes
    assert user_attrs["input.audio.url"].startswith("data:audio/wav;base64,")


def test_realtime_error_event_finalizes_turn_with_error_status(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: Any,
) -> None:
    """A RealtimeError dispatch finalizes the open turn with end_reason=session_closed
    and OTel status ERROR carrying the error message."""

    class _FakeRealtimeError:
        def __init__(self, error: str) -> None:
            self.error = error

    monkeypatch.setattr(_realtime, "_RealtimeError", _FakeRealtimeError)

    state = _state(tracer_provider)
    state.on_speech_started("item-1")
    state.on_user_audio_committed("item-1")
    state.on_user_transcript_completed("item-1", "Hi.")
    state.on_response_created("response-1", "gpt-realtime")

    _dispatch_event(state, _FakeRealtimeError("boom"))

    spans = _spans_by_kind(in_memory_span_exporter)
    audio_spans = spans[_AUDIO_KIND]
    assert len(audio_spans) == 1
    audio = audio_spans[0]
    assert audio.attributes["end_reason"] == "session_closed"
    assert audio.status.status_code == trace_api.StatusCode.ERROR
    assert audio.status.description == "boom"

    # USER and LLM children also closed with ERROR status.
    user_spans = spans[_USER_KIND]
    llm_spans = spans[OpenInferenceSpanKindValues.LLM.value]
    assert len(user_spans) == 1
    assert len(llm_spans) == 1
    assert user_spans[0].status.status_code == trace_api.StatusCode.ERROR
    assert llm_spans[0].status.status_code == trace_api.StatusCode.ERROR


def test_attribute_keys_snapshot_on_full_turn(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Locks in the exact set of attribute keys written to each span kind on a
    fully-populated turn (session config, model, audio + transcripts, tokens,
    TTFT). Any future regression that accidentally adds or drops an attribute
    will fail this test."""
    state = _state(tracer_provider)
    _dispatch_raw(
        state,
        _raw_event(
            {
                "type": "session.created",
                "session": {
                    "id": "sess_abc",
                    "model": "gpt-realtime",
                    "instructions": "Be helpful.",
                    "voice": "alloy",
                },
            }
        ),
    )
    state.on_speech_started("item-1")
    state.on_send_audio(b"\x00\x01" * 256)
    state.on_user_audio_committed("item-1")
    state.on_user_transcript_completed("item-1", "Hi.")
    state.on_response_created("response-1", "gpt-realtime")
    state.on_audio_delta("response-1", b"\x02\x03" * 256)
    state.on_asst_transcript_done("response-1", "Hello!")
    state.on_response_done(
        "response-1",
        usage={
            "input_tokens": 12,
            "output_tokens": 7,
            "total_tokens": 19,
            "input_token_details": {"audio_tokens": 8},
            "output_token_details": {"audio_tokens": 5},
        },
        model_name="gpt-realtime",
    )
    state.on_session_close()

    spans = _spans_by_kind(in_memory_span_exporter)
    audio_attrs = dict(spans[_AUDIO_KIND][0].attributes or {})
    user_attrs = dict(spans[_USER_KIND][0].attributes or {})
    llm_attrs = dict(spans[OpenInferenceSpanKindValues.LLM.value][0].attributes or {})

    assert set(audio_attrs.keys()) == {
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        SpanAttributes.SESSION_ID,
        SpanAttributes.LLM_MODEL_NAME,
        SpanAttributes.LLM_INVOCATION_PARAMETERS,
        SpanAttributes.INPUT_VALUE,
        SpanAttributes.OUTPUT_VALUE,
        "end_reason",
    }
    assert set(user_attrs.keys()) == {
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        SpanAttributes.SESSION_ID,
        "input.audio.url",
        "input.audio.mime_type",
        "input.audio.transcript",
    }
    assert set(llm_attrs.keys()) == {
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        SpanAttributes.SESSION_ID,
        SpanAttributes.LLM_SYSTEM,
        SpanAttributes.LLM_MODEL_NAME,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO,
        "output.audio.url",
        "output.audio.mime_type",
        "output.audio.transcript",
        "time_to_first_token_ms",
    }


# ----------------------------------------------------------------------
# Parent AUDIO span respects redaction flags
# ----------------------------------------------------------------------


def test_hide_inputs_drops_input_value_on_audio_parent(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """TraceConfig(hide_inputs=True) must also suppress input.value on the parent."""
    state = _state(tracer_provider, config=TraceConfig(hide_inputs=True))
    _drive_audio_turn(state)

    audio_attrs = _spans_by_kind(in_memory_span_exporter)[_AUDIO_KIND][0].attributes
    assert audio_attrs is not None
    assert SpanAttributes.INPUT_VALUE not in audio_attrs
    # Output side unaffected.
    assert audio_attrs.get(SpanAttributes.OUTPUT_VALUE) == "Hello!"


def test_hide_outputs_drops_output_value_on_audio_parent(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """TraceConfig(hide_outputs=True) must also suppress output.value on the parent."""
    state = _state(tracer_provider, config=TraceConfig(hide_outputs=True))
    _drive_audio_turn(state)

    audio_attrs = _spans_by_kind(in_memory_span_exporter)[_AUDIO_KIND][0].attributes
    assert audio_attrs is not None
    assert SpanAttributes.OUTPUT_VALUE not in audio_attrs
    assert audio_attrs.get(SpanAttributes.INPUT_VALUE) == "Hi."


def test_hide_input_audio_env_drops_input_value_on_audio_parent(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: Any,
) -> None:
    """OPENINFERENCE_HIDE_INPUT_AUDIO must suppress input.value on the parent too."""
    monkeypatch.setenv("OPENINFERENCE_HIDE_INPUT_AUDIO", "true")
    state = _state(tracer_provider)
    _drive_audio_turn(state)

    audio_attrs = _spans_by_kind(in_memory_span_exporter)[_AUDIO_KIND][0].attributes
    assert audio_attrs is not None
    assert SpanAttributes.INPUT_VALUE not in audio_attrs
    assert audio_attrs.get(SpanAttributes.OUTPUT_VALUE) == "Hello!"


def test_hide_output_audio_env_drops_output_value_on_audio_parent(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: Any,
) -> None:
    """OPENINFERENCE_HIDE_OUTPUT_AUDIO must suppress output.value on the parent too."""
    monkeypatch.setenv("OPENINFERENCE_HIDE_OUTPUT_AUDIO", "true")
    state = _state(tracer_provider)
    _drive_audio_turn(state)

    audio_attrs = _spans_by_kind(in_memory_span_exporter)[_AUDIO_KIND][0].attributes
    assert audio_attrs is not None
    assert SpanAttributes.OUTPUT_VALUE not in audio_attrs
    assert audio_attrs.get(SpanAttributes.INPUT_VALUE) == "Hi."


# ----------------------------------------------------------------------
# Suppress tracing
# ----------------------------------------------------------------------


def test_no_spans_when_tracing_suppressed(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Inside suppress_tracing(), both wrappers must skip instrumentation entirely:
    no session state is created, no spans are produced."""
    tracer = OITracer(
        trace_api.get_tracer(__name__, tracer_provider=tracer_provider),
        config=TraceConfig(),
    )
    put_wrapper = make_realtime_wrapper(tracer, TraceConfig())
    audio_wrapper = make_send_audio_wrapper()

    class _FakeSession:
        pass

    session: Any = _FakeSession()

    async def _orig_put(*args: Any, **kwargs: Any) -> None:
        return None

    async def _orig_send(*args: Any, **kwargs: Any) -> None:
        return None

    fake_event = object()

    with suppress_tracing():
        asyncio.run(put_wrapper(_orig_put, session, (fake_event,), {}))
        asyncio.run(audio_wrapper(_orig_send, session, (b"\x00\x01" * 64,), {}))

    assert in_memory_span_exporter.get_finished_spans() == ()
    assert session not in _session_states
