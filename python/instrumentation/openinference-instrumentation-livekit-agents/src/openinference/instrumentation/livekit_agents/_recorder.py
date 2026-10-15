from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

from openinference.instrumentation.livekit_agents._turns import (
    ModelStep,
    ToolRun,
    Utterance,
    VoiceTurn,
)

_NS = 1_000_000_000


@dataclass
class _OpenTurn:
    start_ns: int
    user: Utterance | None
    steps: list[ModelStep] = field(default_factory=list)


class SessionRecorder:
    """Folds one AgentSession's events into a VoiceTurn per SpeechHandle.

    The user utterance that precedes a speech becomes that turn's user. A transcript that
    arrives after its reply has started, as with preemptive generation, joins the newest
    turn that started after the user began speaking and has no user yet.
    """

    def __init__(self, stt_model: str | None, stt_provider: str | None) -> None:
        self.session_id: str | None = None
        self._stt_model = stt_model
        self._stt_provider = stt_provider
        self._lock = threading.Lock()
        self._speaking_since_ns: int | None = None
        self._last_speech: tuple[int, int] | None = None
        self._pending_user: Utterance | None = None
        self._open: dict[str, _OpenTurn] = {}

    def user_speaking(self, at_ns: int) -> None:
        with self._lock:
            self._speaking_since_ns = at_ns

    def user_listening(self, at_ns: int) -> None:
        with self._lock:
            if self._speaking_since_ns is not None:
                self._last_speech = (self._speaking_since_ns, at_ns)
                self._speaking_since_ns = None

    def transcribed(self, transcript: str, at_ns: int) -> None:
        if not transcript:
            return
        with self._lock:
            start_ns, end_ns = self._last_speech or (self._speaking_since_ns or at_ns, at_ns)
            self._last_speech = None
            utterance = Utterance(transcript, self._stt_model, self._stt_provider, start_ns, end_ns)
            waiting = [
                (turn.start_ns, speech_id)
                for speech_id, turn in self._open.items()
                if turn.user is None and turn.start_ns >= start_ns
            ]
            if waiting:
                self._open[max(waiting)[1]].user = utterance
            else:
                self._pending_user = utterance

    def speech_created(self, speech_id: str, at_ns: int, *, from_user: bool) -> None:
        with self._lock:
            user = self._pending_user if from_user else None
            if from_user:
                self._pending_user = None
            self._open[speech_id] = _OpenTurn(at_ns, user)

    def add_step(self, speech_id: str, step: ModelStep) -> None:
        with self._lock:
            if (turn := self._open.get(speech_id)) is not None:
                turn.steps.append(step)

    def speech_done(
        self, speech_id: str, chat_items: Iterable[Any], interrupted: bool, at_ns: int
    ) -> VoiceTurn | None:
        with self._lock:
            turn = self._open.pop(speech_id, None)
        if turn is None:
            return None
        items = list(chat_items)
        tools = _tool_runs(items)
        replies = [
            item
            for item in items
            if getattr(item, "type", None) == "message" and item.role == "assistant"
        ]
        spoken = " ".join(text for item in replies if (text := item.text_content)) or None
        if turn.user is None and not turn.steps and not tools and spoken is None:
            return None
        interrupted = interrupted or any(item.interrupted for item in replies)
        return VoiceTurn(
            speech_id=speech_id,
            session_id=self.session_id,
            user=turn.user,
            steps=tuple(turn.steps),
            tools=tools,
            spoken_text=spoken,
            end_reason="interrupted" if interrupted else "complete",
            start_ns=turn.user.start_ns if turn.user is not None else turn.start_ns,
            end_ns=at_ns,
        )


def _tool_runs(items: list[Any]) -> tuple[ToolRun, ...]:
    outputs = {
        item.call_id: item
        for item in items
        if getattr(item, "type", None) == "function_call_output"
    }
    runs = []
    for call in items:
        if getattr(call, "type", None) != "function_call":
            continue
        output = outputs.get(call.call_id)
        start_ns = int(call.created_at * _NS)
        runs.append(
            ToolRun(
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
                output=output.output if output is not None else None,
                is_error=bool(output.is_error) if output is not None else False,
                start_ns=start_ns,
                end_ns=int(output.created_at * _NS) if output is not None else start_ns,
            )
        )
    return tuple(runs)
