from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Mapping, Union

EndReason = Literal["complete", "interrupted", "session_closed"]
Phase = Literal["start", "end"]


@dataclass(frozen=True)
class SpanLoc:
    name: str
    parent_id: int | None


@dataclass(frozen=True)
class HeardUtterance:
    """LiveKit does not put PCM on the span."""

    transcript: str
    stt_model: str | None
    stt_provider: str | None
    confidence: float | None
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class ModelStep:
    span_id: int
    source: Literal["llm_request", "realtime_inference"]
    model: str | None
    provider: str | None
    input_messages_json: str | None
    output_messages_json: str | None
    spoken_text: str | None
    input_tokens: int | None
    output_tokens: int | None
    audio_input_tokens: int | None
    audio_output_tokens: int | None
    time_to_first_token_s: float | None
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class ToolCall:
    name: str
    call_id: str | None
    arguments: str | None
    output: str | None
    is_error: bool
    start_ns: int
    end_ns: int
    step_span_id: int | None


@dataclass(frozen=True)
class VoiceTurn:
    trace_id: int
    agent_turn_span_id: int
    speech_id: str | None
    session_id: str | None
    agent_name: str | None
    user: HeardUtterance | None
    typed_text: str | None
    steps: tuple[ModelStep, ...]
    tools: tuple[ToolCall, ...]
    end_reason: EndReason
    failed: bool
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class OpenTurn:
    speech_id: str | None
    session_id: str | None
    agent_name: str | None
    user: HeardUtterance | None
    typed_text: str | None
    steps: tuple[ModelStep, ...]
    tools: tuple[ToolCall, ...]
    interrupted: bool
    failed: bool
    start_ns: int


@dataclass(frozen=True)
class SessionFold:
    session_id: str | None = None
    pending_user: HeardUtterance | None = None
    open_turns: Mapping[int, OpenTurn] = field(default_factory=dict)
    spans: Mapping[int, SpanLoc] = field(default_factory=dict)
    ended: frozenset[int] = frozenset()
    closed: bool = False


@dataclass(frozen=True)
class SpanIndexed:
    trace_id: int
    span_id: int
    parent_id: int | None
    name: str
    phase: Phase


@dataclass(frozen=True)
class UserTurnEnded:
    trace_id: int
    span_id: int
    parent_id: int | None
    name: str
    phase: Phase
    utterance: HeardUtterance | None


@dataclass(frozen=True)
class AgentTurnStarted:
    trace_id: int
    span_id: int
    parent_id: int | None
    name: str
    phase: Phase
    start_ns: int
    speech_id: str | None
    session_id: str | None
    agent_name: str | None


@dataclass(frozen=True)
class AgentTurnEnded:
    trace_id: int
    span_id: int
    parent_id: int | None
    name: str
    phase: Phase
    end_ns: int
    session_id: str | None
    speech_id: str | None
    agent_name: str | None
    typed_text: str | None
    interrupted: bool
    failed: bool


@dataclass(frozen=True)
class ModelStepEnded:
    trace_id: int
    span_id: int
    parent_id: int | None
    name: str
    phase: Phase
    step: ModelStep


@dataclass(frozen=True)
class ToolEnded:
    trace_id: int
    span_id: int
    parent_id: int | None
    name: str
    phase: Phase
    tool: ToolCall


@dataclass(frozen=True)
class SpokenTextEnded:
    trace_id: int
    span_id: int
    parent_id: int | None
    name: str
    phase: Phase
    text: str
    start_ns: int


@dataclass(frozen=True)
class StepLatencyNoted:
    trace_id: int
    span_id: int
    parent_id: int | None
    name: str
    phase: Phase
    time_to_first_token_s: float


@dataclass(frozen=True)
class AudioUsageNoted:
    trace_id: int
    span_id: int
    parent_id: int | None
    name: str
    phase: Phase
    audio_input_tokens: int | None
    audio_output_tokens: int | None
    time_to_first_token_s: float | None


@dataclass(frozen=True)
class SessionEnded:
    trace_id: int
    span_id: int
    parent_id: int | None
    name: str
    phase: Phase


TraceEvent = Union[
    SpanIndexed,
    UserTurnEnded,
    AgentTurnStarted,
    AgentTurnEnded,
    ModelStepEnded,
    ToolEnded,
    SpokenTextEnded,
    StepLatencyNoted,
    AudioUsageNoted,
    SessionEnded,
]


def reduce_trace(
    state: SessionFold,
    event: TraceEvent,
) -> tuple[SessionFold, tuple[VoiceTurn, ...]]:
    if event.span_id in state.ended:
        return state, ()
    if event.phase == "end" and event.span_id not in state.spans:
        return state, ()

    spans = dict(state.spans)
    ended = state.ended
    agent_turn_span_id: int | None = None
    if event.phase == "start":
        spans[event.span_id] = SpanLoc(event.name, event.parent_id)
    else:
        # Leave the location. The child ends first, and the later llm_node end
        # still walks from that child.
        agent_turn_span_id = _agent_turn_span_id(spans, event.parent_id)
        ended = ended | {event.span_id}

    event_session_id = getattr(event, "session_id", None)
    session_id = state.session_id if state.session_id is not None else event_session_id
    next_state = replace(state, session_id=session_id, spans=spans, ended=ended)

    if isinstance(event, SpanIndexed):
        return next_state, ()

    if isinstance(event, UserTurnEnded):
        if event.utterance is None:
            return next_state, ()
        candidates = [
            (span_id, turn)
            for span_id, turn in next_state.open_turns.items()
            if turn.user is None and turn.start_ns >= event.utterance.start_ns
        ]
        if not candidates:
            return replace(next_state, pending_user=event.utterance), ()
        span_id, turn = min(candidates, key=lambda item: item[1].start_ns)
        open_turns = dict(next_state.open_turns)
        open_turns[span_id] = replace(turn, user=event.utterance)
        return replace(next_state, open_turns=open_turns), ()

    if isinstance(event, AgentTurnStarted):
        open_turns = dict(next_state.open_turns)
        open_turns[event.span_id] = OpenTurn(
            speech_id=event.speech_id,
            session_id=event.session_id,
            agent_name=event.agent_name,
            user=next_state.pending_user,
            typed_text=None,
            steps=(),
            tools=(),
            interrupted=False,
            failed=False,
            start_ns=event.start_ns,
        )
        return replace(next_state, pending_user=None, open_turns=open_turns), ()

    if isinstance(event, ModelStepEnded):
        if agent_turn_span_id is None or agent_turn_span_id not in next_state.open_turns:
            return next_state, ()
        open_turns = dict(next_state.open_turns)
        turn = open_turns[agent_turn_span_id]
        open_turns[agent_turn_span_id] = replace(turn, steps=turn.steps + (event.step,))
        return replace(next_state, open_turns=open_turns), ()

    if isinstance(event, ToolEnded):
        if agent_turn_span_id is None or agent_turn_span_id not in next_state.open_turns:
            return next_state, ()
        open_turns = dict(next_state.open_turns)
        turn = open_turns[agent_turn_span_id]
        eligible_indexes = [
            index for index, step in enumerate(turn.steps) if step.start_ns <= event.tool.start_ns
        ]
        step_span_id = None
        if eligible_indexes:
            step_index = max(
                eligible_indexes,
                key=lambda index: (turn.steps[index].start_ns, index),
            )
            step_span_id = turn.steps[step_index].span_id
        tool = replace(event.tool, step_span_id=step_span_id)
        open_turns[agent_turn_span_id] = replace(turn, tools=turn.tools + (tool,))
        return replace(next_state, open_turns=open_turns), ()

    if isinstance(event, SpokenTextEnded):
        if agent_turn_span_id is None or agent_turn_span_id not in next_state.open_turns:
            return next_state, ()
        open_turns = dict(next_state.open_turns)
        turn = open_turns[agent_turn_span_id]
        eligible_indexes = [
            index for index, step in enumerate(turn.steps) if step.start_ns <= event.start_ns
        ]
        if not eligible_indexes:
            return next_state, ()
        step_index = max(
            eligible_indexes,
            key=lambda index: (turn.steps[index].start_ns, index),
        )
        steps = list(turn.steps)
        steps[step_index] = replace(steps[step_index], spoken_text=event.text)
        open_turns[agent_turn_span_id] = replace(turn, steps=tuple(steps))
        return replace(next_state, open_turns=open_turns), ()

    if isinstance(event, StepLatencyNoted):
        open_turns = dict(next_state.open_turns)
        changed = False
        for turn_span_id, turn in next_state.open_turns.items():
            latency_steps = tuple(
                replace(step, time_to_first_token_s=event.time_to_first_token_s)
                if step.time_to_first_token_s is None
                and _walk_reaches(next_state.spans, step.span_id, event.span_id)
                else step
                for step in turn.steps
            )
            if latency_steps != turn.steps:
                open_turns[turn_span_id] = replace(turn, steps=latency_steps)
                changed = True
        return (replace(next_state, open_turns=open_turns) if changed else next_state), ()

    if isinstance(event, AudioUsageNoted):
        target_span_ids = {event.span_id, event.parent_id}
        open_turns = dict(next_state.open_turns)
        for turn_span_id, turn in next_state.open_turns.items():
            for step_index, step in enumerate(turn.steps):
                if step.span_id not in target_span_ids:
                    continue
                audio_steps = list(turn.steps)
                audio_steps[step_index] = replace(
                    step,
                    audio_input_tokens=(
                        event.audio_input_tokens
                        if step.audio_input_tokens is None
                        else step.audio_input_tokens
                    ),
                    audio_output_tokens=(
                        event.audio_output_tokens
                        if step.audio_output_tokens is None
                        else step.audio_output_tokens
                    ),
                    time_to_first_token_s=(
                        event.time_to_first_token_s
                        if step.time_to_first_token_s is None
                        else step.time_to_first_token_s
                    ),
                )
                open_turns[turn_span_id] = replace(turn, steps=tuple(audio_steps))
                return replace(next_state, open_turns=open_turns), ()
        return next_state, ()

    if isinstance(event, AgentTurnEnded):
        ended_turn = next_state.open_turns.get(event.span_id)
        if ended_turn is None:
            return next_state, ()
        ended_turn = replace(
            ended_turn,
            speech_id=ended_turn.speech_id or event.speech_id,
            session_id=ended_turn.session_id or event.session_id,
            agent_name=ended_turn.agent_name or event.agent_name,
            typed_text=ended_turn.typed_text or event.typed_text,
            interrupted=ended_turn.interrupted or event.interrupted,
            failed=ended_turn.failed or event.failed,
        )
        sealed = _seal(
            next_state,
            trace_id=event.trace_id,
            agent_turn_span_id=event.span_id,
            turn=ended_turn,
            end_ns=event.end_ns,
            end_reason="interrupted" if ended_turn.interrupted else "complete",
        )
        open_turns = dict(next_state.open_turns)
        open_turns.pop(event.span_id)
        next_state = replace(next_state, open_turns=open_turns)
        return next_state, (sealed,) if sealed is not None else ()

    if isinstance(event, SessionEnded):
        turns = []
        for turn_span_id, turn in next_state.open_turns.items():
            end_ns = max((step.end_ns for step in turn.steps), default=turn.start_ns)
            end_ns = max(end_ns, *(tool.end_ns for tool in turn.tools))
            sealed = _seal(
                next_state,
                trace_id=event.trace_id,
                agent_turn_span_id=turn_span_id,
                turn=turn,
                end_ns=end_ns,
                end_reason="session_closed",
            )
            if sealed is not None:
                turns.append(sealed)
        closed_state = SessionFold(
            session_id=next_state.session_id,
            ended=next_state.ended,
            closed=True,
        )
        return closed_state, tuple(turns)

    return next_state, ()


def _agent_turn_span_id(spans: Mapping[int, SpanLoc], span_id: int | None) -> int | None:
    seen: set[int] = set()
    while span_id is not None and span_id not in seen:
        seen.add(span_id)
        location = spans.get(span_id)
        if location is None:
            return None
        if location.name == "agent_turn":
            return span_id
        span_id = location.parent_id
    return None


def _walk_reaches(spans: Mapping[int, SpanLoc], span_id: int, target_id: int) -> bool:
    seen: set[int] = set()
    while span_id not in seen:
        if span_id == target_id:
            return True
        seen.add(span_id)
        location = spans.get(span_id)
        if location is None or location.parent_id is None:
            return False
        span_id = location.parent_id
    return False


def _seal(
    state: SessionFold,
    *,
    trace_id: int,
    agent_turn_span_id: int,
    turn: OpenTurn,
    end_ns: int,
    end_reason: EndReason,
) -> VoiceTurn | None:
    if turn.user is None and not turn.steps and not turn.tools:
        return None
    return VoiceTurn(
        trace_id=trace_id,
        agent_turn_span_id=agent_turn_span_id,
        speech_id=turn.speech_id,
        session_id=turn.session_id or state.session_id,
        agent_name=turn.agent_name,
        user=turn.user,
        typed_text=turn.typed_text,
        steps=turn.steps,
        tools=turn.tools,
        end_reason=end_reason,
        failed=turn.failed,
        start_ns=turn.user.start_ns if turn.user is not None else turn.start_ns,
        end_ns=end_ns,
    )
