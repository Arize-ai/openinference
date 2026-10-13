from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from openinference.instrumentation import Message

EndReason = Literal["complete", "interrupted"]


@dataclass(frozen=True)
class Utterance:
    """What the user said, as transcribed, between VAD start and end of speech."""

    transcript: str
    stt_model: str | None
    stt_provider: str | None
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class RequestedToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelStep:
    """One LLMStream: what the model was sent and what it streamed back."""

    model: str | None
    provider: str | None
    input_messages: tuple[Message, ...]
    output_text: str
    tool_calls: tuple[RequestedToolCall, ...]
    input_tokens: int | None
    output_tokens: int | None
    time_to_first_token_s: float | None
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class ToolRun:
    call_id: str
    name: str
    arguments: str
    output: str | None
    is_error: bool
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class VoiceTurn:
    """One SpeechHandle, from the user's speech to the end of the agent's reply."""

    speech_id: str
    session_id: str | None
    user: Utterance | None
    steps: tuple[ModelStep, ...]
    tools: tuple[ToolRun, ...]
    spoken_text: str | None
    end_reason: EndReason
    start_ns: int
    end_ns: int

    def step_for(self, tool: ToolRun) -> ModelStep | None:
        return next(
            (
                step
                for step in self.steps
                if any(call.call_id == tool.call_id for call in step.tool_calls)
            ),
            None,
        )
