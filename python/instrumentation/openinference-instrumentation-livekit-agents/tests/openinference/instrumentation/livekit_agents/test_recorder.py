from dataclasses import dataclass

from openinference.instrumentation.livekit_agents._recorder import SessionRecorder
from openinference.instrumentation.livekit_agents._turns import (
    ModelStep,
    RequestedToolCall,
    ToolRun,
    Utterance,
)

_S = 1_000_000_000


@dataclass
class _Message:
    role: str
    text_content: str | None
    interrupted: bool = False
    type: str = "message"


@dataclass
class _Call:
    call_id: str
    name: str
    arguments: str
    created_at: float
    type: str = "function_call"


@dataclass
class _Output:
    call_id: str
    output: str
    is_error: bool
    created_at: float
    type: str = "function_call_output"


def _step(start: int, *calls: RequestedToolCall) -> ModelStep:
    return ModelStep(None, None, (), "", calls, None, None, None, start * _S, (start + 1) * _S)


def _spoken(recorder: SessionRecorder) -> None:
    recorder.user_speaking(1 * _S)
    recorder.user_listening(3 * _S)


def test_user_utterance_and_tool_belong_to_the_reply() -> None:
    recorder = SessionRecorder("whisper-1", "openai")
    _spoken(recorder)
    recorder.transcribed("weather?", 4 * _S)
    recorder.speech_created("sp1", 4 * _S, from_user=True)
    asks = _step(5, RequestedToolCall("c1", "get_weather", "{}"))
    recorder.add_step("sp1", asks)
    items = [_Call("c1", "get_weather", "{}", 6.0), _Output("c1", "sunny", False, 7.0)]
    turn = recorder.speech_done(
        "sp1", [*items, _Message("assistant", "It is sunny.")], False, 9 * _S
    )
    assert turn is not None
    assert turn.user == Utterance("weather?", "whisper-1", "openai", 1 * _S, 3 * _S)
    assert turn.tools == (ToolRun("c1", "get_weather", "{}", "sunny", False, 6 * _S, 7 * _S),)
    assert turn.step_for(turn.tools[0]) is asks
    assert (turn.spoken_text, turn.end_reason) == ("It is sunny.", "complete")
    assert (turn.start_ns, turn.end_ns) == (1 * _S, 9 * _S)


def test_a_late_transcript_joins_the_reply_already_under_way() -> None:
    recorder = SessionRecorder(None, None)
    _spoken(recorder)
    recorder.speech_created("sp1", 3 * _S, from_user=True)
    recorder.transcribed("hello", 4 * _S)
    turn = recorder.speech_done("sp1", [_Message("assistant", "Hi.")], False, 6 * _S)
    assert turn is not None and turn.user is not None
    assert turn.user.transcript == "hello"


def test_say_does_not_take_the_users_utterance() -> None:
    recorder = SessionRecorder(None, None)
    _spoken(recorder)
    recorder.transcribed("hello", 4 * _S)
    recorder.speech_created("greeting", 4 * _S, from_user=False)
    recorder.speech_created("reply", 5 * _S, from_user=True)
    greeting = recorder.speech_done("greeting", [_Message("assistant", "Welcome.")], False, 6 * _S)
    reply = recorder.speech_done("reply", [_Message("assistant", "Hi.")], False, 7 * _S)
    assert greeting is not None and greeting.user is None
    assert reply is not None and reply.user is not None and reply.user.transcript == "hello"


def test_barge_in_keeps_only_what_was_heard() -> None:
    recorder = SessionRecorder(None, None)
    recorder.speech_created("sp1", 1 * _S, from_user=True)
    heard = _Message("assistant", "Once upon a", interrupted=True)
    turn = recorder.speech_done("sp1", [heard], False, 2 * _S)
    assert turn is not None
    assert (turn.spoken_text, turn.end_reason) == ("Once upon a", "interrupted")


def test_a_tool_cut_off_before_it_returned_has_no_output() -> None:
    recorder = SessionRecorder(None, None)
    recorder.speech_created("sp1", 1 * _S, from_user=True)
    turn = recorder.speech_done("sp1", [_Call("c1", "slow", "{}", 2.0)], True, 3 * _S)
    assert turn is not None
    assert turn.tools == (ToolRun("c1", "slow", "{}", None, False, 2 * _S, 2 * _S),)
    assert turn.end_reason == "interrupted"


def test_an_empty_or_unknown_speech_renders_nothing() -> None:
    recorder = SessionRecorder(None, None)
    recorder.speech_created("sp1", 1 * _S, from_user=True)
    assert recorder.speech_done("sp1", [], True, 2 * _S) is None
    assert (
        recorder.speech_done("never-created", [_Message("assistant", "x")], False, 2 * _S) is None
    )
    recorder.add_step("never-created", _step(1))
