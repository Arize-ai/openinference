from openinference.instrumentation.livekit_agents._fold import (
    AgentTurnEnded,
    AgentTurnStarted,
    HeardUtterance,
    ModelStep,
    ModelStepEnded,
    SessionEnded,
    SessionFold,
    SpanIndexed,
    SpokenTextEnded,
    StepLatencyNoted,
    ToolCall,
    ToolEnded,
    UserTurnEnded,
    VoiceTurn,
    reduce_trace,
)


def _start(
    state: SessionFold,
    *,
    span_id: int,
    name: str,
    parent_id: int | None,
) -> SessionFold:
    state, turns = reduce_trace(
        state,
        SpanIndexed(1, span_id, parent_id, name, "start"),
    )
    assert turns == ()
    return state


def _start_agent(
    state: SessionFold,
    *,
    span_id: int,
    start_ns: int,
    session_id: str | None = None,
) -> SessionFold:
    state, turns = reduce_trace(
        state,
        AgentTurnStarted(
            trace_id=1,
            span_id=span_id,
            parent_id=None,
            name="agent_turn",
            phase="start",
            start_ns=start_ns,
            speech_id=None,
            session_id=session_id,
            agent_name=None,
        ),
    )
    assert turns == ()
    return state


def _end_agent(
    state: SessionFold,
    *,
    span_id: int,
    end_ns: int,
    interrupted: bool = False,
    speech_id: str | None = None,
    session_id: str | None = None,
) -> tuple[SessionFold, tuple[VoiceTurn, ...]]:
    return reduce_trace(
        state,
        AgentTurnEnded(
            trace_id=1,
            span_id=span_id,
            parent_id=None,
            name="agent_turn",
            phase="end",
            end_ns=end_ns,
            session_id=session_id,
            speech_id=speech_id,
            agent_name=None,
            typed_text=None,
            interrupted=interrupted,
            failed=False,
        ),
    )


def _utterance(
    state: SessionFold,
    *,
    span_id: int,
    transcript: str | None,
    start_ns: int,
    end_ns: int,
) -> SessionFold:
    state = _start(state, span_id=span_id, name="user_turn", parent_id=None)
    heard = (
        HeardUtterance(
            transcript=transcript,
            stt_model=None,
            stt_provider=None,
            confidence=None,
            start_ns=start_ns,
            end_ns=end_ns,
        )
        if transcript is not None
        else None
    )
    state, turns = reduce_trace(
        state,
        UserTurnEnded(1, span_id, None, "user_turn", "end", heard),
    )
    assert turns == ()
    return state


def _model_step(
    state: SessionFold,
    *,
    span_id: int,
    parent_id: int,
    start_ns: int,
    end_ns: int,
    model: str = "gpt",
) -> SessionFold:
    state = _start(state, span_id=span_id, name="llm_request", parent_id=parent_id)
    step = ModelStep(
        span_id=span_id,
        source="llm_request",
        model=model,
        provider=None,
        input_messages_json='{"role":"user"}',
        output_messages_json='{"role":"assistant"}',
        spoken_text=None,
        input_tokens=None,
        output_tokens=None,
        audio_input_tokens=None,
        audio_output_tokens=None,
        time_to_first_token_s=None,
        start_ns=start_ns,
        end_ns=end_ns,
    )
    state, turns = reduce_trace(
        state,
        ModelStepEnded(1, span_id, parent_id, "llm_request", "end", step),
    )
    assert turns == ()
    return state


def test_cascaded_tool_turn() -> None:
    state = _utterance(
        SessionFold(),
        span_id=10,
        transcript="hello",
        start_ns=0,
        end_ns=100,
    )
    state = _start_agent(state, span_id=20, start_ns=110, session_id="RM_room")
    state = _start(state, span_id=30, name="llm_node", parent_id=20)
    state = _model_step(state, span_id=31, parent_id=30, start_ns=120, end_ns=180)

    state = _start(state, span_id=40, name="function_tool", parent_id=20)
    state, turns = reduce_trace(
        state,
        ToolEnded(
            trace_id=1,
            span_id=40,
            parent_id=20,
            name="function_tool",
            phase="end",
            tool=ToolCall(
                name="lookup",
                call_id=None,
                arguments='{"q":1}',
                output="ok",
                is_error=False,
                start_ns=150,
                end_ns=170,
                step_span_id=None,
            ),
        ),
    )
    assert turns == ()

    state = _start(state, span_id=49, name="tts_node", parent_id=20)
    state = _start(state, span_id=50, name="tts_request", parent_id=49)
    state, turns = reduce_trace(
        state,
        SpokenTextEnded(1, 50, 49, "tts_request", "end", "hi there", 160),
    )
    assert turns == ()

    state, turns = _end_agent(
        state,
        span_id=20,
        end_ns=200,
        speech_id="sp-1",
        session_id="RM_room",
    )

    assert len(turns) == 1
    turn = turns[0]
    assert turn.user is not None
    assert turn.user.transcript == "hello"
    assert len(turn.steps) == 1
    assert turn.steps[0].spoken_text == "hi there"
    assert turn.steps[0].input_messages_json == '{"role":"user"}'
    assert turn.steps[0].output_messages_json == '{"role":"assistant"}'
    assert turn.steps[0].model == "gpt"
    assert len(turn.tools) == 1
    assert turn.tools[0].name == "lookup"
    assert turn.tools[0].output == "ok"
    assert turn.tools[0].step_span_id == turn.steps[0].span_id
    assert turn.session_id == "RM_room"
    assert turn.speech_id == "sp-1"
    assert turn.end_reason == "complete"
    assert turn.start_ns == 0
    assert turn.end_ns == 200


def test_dropped_user_turn_does_not_fill_agent_user() -> None:
    state = _utterance(
        SessionFold(),
        span_id=10,
        transcript=None,
        start_ns=0,
        end_ns=100,
    )
    state = _start_agent(state, span_id=20, start_ns=110)
    state = _model_step(state, span_id=30, parent_id=20, start_ns=120, end_ns=180)
    _, turns = _end_agent(state, span_id=20, end_ns=200)

    assert len(turns) == 1
    assert turns[0].user is None
    assert turns[0].steps[0].span_id == 30
    assert turns[0].start_ns == 110


def test_preemptive_overlap_attaches_user_to_open_turn() -> None:
    state = _start_agent(SessionFold(), span_id=20, start_ns=50)
    state = _utterance(
        state,
        span_id=10,
        transcript="still talking",
        start_ns=0,
        end_ns=100,
    )
    _, turns = _end_agent(state, span_id=20, end_ns=150)

    assert len(turns) == 1
    assert turns[0].user is not None
    assert turns[0].user.transcript == "still talking"
    assert turns[0].start_ns == 0


def test_barge_in_keeps_first_user_and_moves_second_to_next_turn() -> None:
    state = _start_agent(SessionFold(), span_id=20, start_ns=50)
    state = _utterance(state, span_id=10, transcript="one", start_ns=0, end_ns=60)
    state = _utterance(state, span_id=11, transcript="two", start_ns=75, end_ns=100)

    state, first_turns = _end_agent(
        state,
        span_id=20,
        end_ns=110,
        interrupted=True,
    )
    state = _start_agent(state, span_id=21, start_ns=120)
    state = _model_step(state, span_id=30, parent_id=21, start_ns=130, end_ns=180)
    _, second_turns = _end_agent(state, span_id=21, end_ns=200)

    assert len(first_turns) == 1
    assert first_turns[0].user is not None
    assert first_turns[0].user.transcript == "one"
    assert first_turns[0].end_reason == "interrupted"
    assert len(second_turns) == 1
    assert second_turns[0].user is not None
    assert second_turns[0].user.transcript == "two"
    assert second_turns[0].steps[0].span_id == 30
    assert second_turns[0].end_reason == "complete"


def test_empty_agent_turn_is_omitted_but_model_turn_is_returned() -> None:
    state = _start_agent(SessionFold(), span_id=20, start_ns=10)
    state, empty_turns = _end_agent(state, span_id=20, end_ns=20)
    state = _start_agent(state, span_id=21, start_ns=30)
    state = _model_step(state, span_id=30, parent_id=21, start_ns=40, end_ns=50)
    _, model_turns = _end_agent(state, span_id=21, end_ns=60)

    assert empty_turns == ()
    assert len(model_turns) == 1
    assert model_turns[0].user is None
    assert model_turns[0].steps[0].span_id == 30
    assert model_turns[0].start_ns == 30
    assert model_turns[0].end_ns == 60


def test_llm_node_latency_fills_its_request_after_the_request_ended() -> None:
    state = _start_agent(SessionFold(), span_id=20, start_ns=10)
    state = _start(state, span_id=30, name="llm_node", parent_id=20)
    state = _model_step(state, span_id=31, parent_id=30, start_ns=20, end_ns=40)
    state = _start(state, span_id=32, name="llm_node", parent_id=20)
    state = _model_step(state, span_id=33, parent_id=32, start_ns=50, end_ns=70, model="other")
    state, turns = reduce_trace(
        state,
        StepLatencyNoted(1, 30, 20, "llm_node", "end", 0.25),
    )
    assert turns == ()
    _, sealed = _end_agent(state, span_id=20, end_ns=80)

    assert len(sealed) == 1
    by_model = {step.model: step.time_to_first_token_s for step in sealed[0].steps}
    assert by_model == {"gpt": 0.25, "other": None}


def test_session_close_without_tools_seals_the_open_turn() -> None:
    state = _start(SessionFold(), span_id=1, name="agent_session", parent_id=None)
    state = _start_agent(state, span_id=20, start_ns=10, session_id="RM_room")
    state = _start(state, span_id=30, name="llm_node", parent_id=20)
    state = _model_step(state, span_id=31, parent_id=30, start_ns=20, end_ns=40)
    _, turns = reduce_trace(state, SessionEnded(1, 1, None, "agent_session", "end"))

    assert len(turns) == 1
    assert turns[0].end_reason == "session_closed"
    assert turns[0].steps[0].span_id == 31
    assert turns[0].tools == ()
    assert turns[0].end_ns == 40
    assert turns[0].session_id == "RM_room"


def test_tool_that_ends_before_its_request_parents_to_that_request() -> None:
    state = _start_agent(SessionFold(), span_id=20, start_ns=10)
    state = _start(state, span_id=30, name="llm_node", parent_id=20)
    state = _start(state, span_id=31, name="llm_request", parent_id=30)
    state = _start(state, span_id=40, name="function_tool", parent_id=20)
    state, turns = reduce_trace(
        state,
        ToolEnded(
            1,
            40,
            20,
            "function_tool",
            "end",
            ToolCall("lookup", None, "{}", "ok", False, 25, 35, None),
        ),
    )
    assert turns == ()
    step = ModelStep(
        31,
        "llm_request",
        "gpt",
        None,
        None,
        None,
        "hi",
        3,
        5,
        None,
        None,
        1.5,
        20,
        50,
    )
    state, turns = reduce_trace(state, ModelStepEnded(1, 31, 30, "llm_request", "end", step))
    assert turns == ()
    _, sealed = _end_agent(state, span_id=20, end_ns=60)

    assert len(sealed) == 1
    assert sealed[0].tools[0].name == "lookup"
    assert sealed[0].tools[0].step_span_id == 31


def test_two_synthesized_sentences_both_stay_on_the_step() -> None:
    state = _start_agent(SessionFold(), span_id=20, start_ns=10)
    state = _start(state, span_id=30, name="llm_node", parent_id=20)
    state = _model_step(state, span_id=31, parent_id=30, start_ns=20, end_ns=40)
    state = _start(state, span_id=49, name="tts_node", parent_id=20)
    state = _start(state, span_id=50, name="tts_request", parent_id=49)
    state, turns = reduce_trace(
        state,
        SpokenTextEnded(1, 50, 49, "tts_request", "end", "Once upon a time.", 45),
    )
    assert turns == ()
    state = _start(state, span_id=51, name="tts_request", parent_id=49)
    state, turns = reduce_trace(
        state,
        SpokenTextEnded(1, 51, 49, "tts_request", "end", "The story of London begins.", 55),
    )
    assert turns == ()
    _, sealed = _end_agent(state, span_id=20, end_ns=80)

    assert len(sealed) == 1
    assert sealed[0].steps[0].spoken_text == "Once upon a time. The story of London begins."


def test_duplicate_agent_turn_end_does_not_emit_twice() -> None:
    state = _start_agent(SessionFold(), span_id=20, start_ns=10)
    state = _model_step(state, span_id=30, parent_id=20, start_ns=20, end_ns=30)
    state, first_turns = _end_agent(state, span_id=20, end_ns=40)
    duplicate_state, duplicate_turns = _end_agent(state, span_id=20, end_ns=40)

    assert len(first_turns) == 1
    assert first_turns[0].agent_turn_span_id == 20
    assert first_turns[0].steps[0].span_id == 30
    assert first_turns[0].end_ns == 40
    assert duplicate_state is state
    assert duplicate_turns == ()
