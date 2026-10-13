from __future__ import annotations

import asyncio
import logging
import time
import weakref
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.livekit_agents._audio import SessionAudio
from openinference.instrumentation.livekit_agents._messages import oi_messages
from openinference.instrumentation.livekit_agents._recorder import SessionRecorder
from openinference.instrumentation.livekit_agents._render import render_turn
from openinference.instrumentation.livekit_agents._taps import attach_audio_taps
from openinference.instrumentation.livekit_agents._turns import ModelStep, RequestedToolCall
from openinference.semconv.trace import OpenInferenceLLMProviderValues

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# LiveKit sets this around every speech task, so a stream created inside one knows its turn.
_speech_handle_var: ContextVar[Any] | None
try:
    from livekit.agents.voice.agent_activity import _SpeechHandleContextVar

    _speech_handle_var = _SpeechHandleContextVar
except ImportError:  # pragma: no cover
    _speech_handle_var = None
    logger.warning("livekit-agents has no _SpeechHandleContextVar; LLM calls stay unattributed")

# Some LiveKit plugins report the API host as their provider.
_PROVIDER_HOSTS = {
    "api.openai.com": OpenInferenceLLMProviderValues.OPENAI.value,
    "api.anthropic.com": OpenInferenceLLMProviderValues.ANTHROPIC.value,
    "api.groq.com": OpenInferenceLLMProviderValues.GROQ.value,
    "api.deepseek.com": OpenInferenceLLMProviderValues.DEEPSEEK.value,
    "api.mistral.ai": OpenInferenceLLMProviderValues.MISTRALAI.value,
    "api.x.ai": OpenInferenceLLMProviderValues.XAI.value,
    "api.cerebras.ai": OpenInferenceLLMProviderValues.CEREBRAS.value,
    "api.fireworks.ai": OpenInferenceLLMProviderValues.FIREWORKS.value,
    "api.together.xyz": OpenInferenceLLMProviderValues.TOGETHER.value,
    "generativelanguage.googleapis.com": OpenInferenceLLMProviderValues.GOOGLE.value,
}


def provider_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value or value == "unknown":
        return None
    return _PROVIDER_HOSTS.get(value.lower(), value)


def _model_name(value: Any) -> str | None:
    return value if isinstance(value, str) and value and value != "unknown" else None


def _suppressed() -> bool:
    return bool(context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY))


@dataclass
class _Session:
    recorder: SessionRecorder
    audio: SessionAudio
    tracer: OITracer
    config: TraceConfig

    def finish(self, handle: Any) -> None:
        try:
            turn = self.recorder.speech_done(
                handle.id, handle.chat_items, handle.interrupted, time.time_ns()
            )
            if turn is not None:
                render_turn(turn, self.tracer, self.config, self.audio)
        except Exception:
            logger.exception("Failed to record LiveKit voice turn")


_SESSIONS_BY_SPEECH: weakref.WeakKeyDictionary[Any, _Session] = weakref.WeakKeyDictionary()


class StartWrapper:
    def __init__(self, tracer: OITracer, config: TraceConfig) -> None:
        self._tracer = tracer
        self._config = config

    async def __call__(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if _suppressed():
            return await wrapped(*args, **kwargs)
        try:
            stt = instance.stt
            session = _Session(
                SessionRecorder(
                    _model_name(getattr(stt, "model", None)),
                    provider_name(getattr(stt, "provider", None)),
                ),
                SessionAudio(),
                self._tracer,
                self._config,
            )
            _subscribe(instance, session)
        except Exception:
            logger.exception("Failed to observe LiveKit AgentSession")
            return await wrapped(*args, **kwargs)
        result = await wrapped(*args, **kwargs)
        try:
            attach_audio_taps(instance, session.audio)
            session.recorder.session_id = await _room_sid(kwargs.get("room"))
        except Exception:
            logger.exception("Failed to finish observing LiveKit AgentSession")
        return result


def _subscribe(instance: Any, session: _Session) -> None:
    recorder = session.recorder

    def on_user_state(event: Any) -> None:
        at_ns = int(event.created_at * 1_000_000_000)
        if event.new_state == "speaking":
            recorder.user_speaking(at_ns)
        elif event.old_state == "speaking":
            recorder.user_listening(at_ns)

    def on_transcribed(event: Any) -> None:
        if event.is_final:
            recorder.transcribed(event.transcript, int(event.created_at * 1_000_000_000))

    def on_speech(event: Any) -> None:
        handle = event.speech_handle
        _SESSIONS_BY_SPEECH[handle] = session
        recorder.speech_created(handle.id, time.time_ns(), from_user=event.source != "say")
        handle.add_done_callback(session.finish)

    instance.on("user_state_changed", _guarded(on_user_state))
    instance.on("user_input_transcribed", _guarded(on_transcribed))
    instance.on("speech_created", _guarded(on_speech))


def _guarded(handler: Callable[[Any], None]) -> Callable[[Any], None]:
    def guarded(event: Any) -> None:
        try:
            handler(event)
        except Exception:
            logger.exception("Failed to record LiveKit session event")

    return guarded


async def _room_sid(room: Any) -> str | None:
    from livekit.agents import get_job_context

    job = get_job_context(required=False)
    if job is not None and job.job.room.sid:
        return str(job.job.room.sid)
    if room is None:
        return None
    try:
        return str(await asyncio.wait_for(room.sid, timeout=2)) or None
    except Exception:
        return None


@dataclass
class _StreamRecord:
    session: _Session
    speech_id: str
    model: str | None
    provider: str | None
    input_items: list[Any]
    start_ns: int
    first_chunk_ns: int | None = None
    text: list[str] = field(default_factory=list)
    tool_calls: list[RequestedToolCall] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    done: bool = False

    def add(self, chunk: Any) -> None:
        delta = getattr(chunk, "delta", None)
        if delta is not None and (delta.content or delta.tool_calls):
            if self.first_chunk_ns is None:
                self.first_chunk_ns = time.time_ns()
            if delta.content:
                self.text.append(delta.content)
            for call in delta.tool_calls or ():
                self.tool_calls.append(RequestedToolCall(call.call_id, call.name, call.arguments))
        if (usage := getattr(chunk, "usage", None)) is not None:
            self.input_tokens = usage.prompt_tokens
            self.output_tokens = usage.completion_tokens

    def finish(self) -> None:
        if self.done:
            return
        self.done = True
        ttft = (self.first_chunk_ns - self.start_ns) / 1e9 if self.first_chunk_ns else None
        self.session.recorder.add_step(
            self.speech_id,
            ModelStep(
                model=self.model,
                provider=self.provider,
                input_messages=tuple(oi_messages(self.input_items)),
                output_text="".join(self.text),
                tool_calls=tuple(self.tool_calls),
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                time_to_first_token_s=ttft,
                start_ns=self.start_ns,
                end_ns=time.time_ns(),
            ),
        )


_STREAMS: weakref.WeakKeyDictionary[Any, _StreamRecord] = weakref.WeakKeyDictionary()


def llm_stream_init(
    wrapped: Callable[..., None], instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    wrapped(*args, **kwargs)
    if _speech_handle_var is None:
        return
    try:
        handle = _speech_handle_var.get(None)
        if handle is None or (session := _SESSIONS_BY_SPEECH.get(handle)) is None:
            return
        llm = instance._llm
        _STREAMS[instance] = _StreamRecord(
            session=session,
            speech_id=handle.id,
            model=_model_name(getattr(llm, "model", None)),
            provider=provider_name(getattr(llm, "provider", None)),
            input_items=list(instance.chat_ctx.items),
            start_ns=time.time_ns(),
        )
    except Exception:
        logger.exception("Failed to observe LiveKit LLMStream")


async def llm_stream_anext(
    wrapped: Callable[..., Awaitable[Any]],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    record = _STREAMS.get(instance)
    try:
        chunk = await wrapped(*args, **kwargs)
    except BaseException:
        if record is not None:
            _safely(record.finish)
        raise
    if record is not None:
        _safely(lambda: record.add(chunk))
    return chunk


async def llm_stream_aclose(
    wrapped: Callable[..., Awaitable[Any]],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    if (record := _STREAMS.get(instance)) is not None:
        _safely(record.finish)
    return await wrapped(*args, **kwargs)


def _safely(action: Callable[[], None]) -> None:
    try:
        action()
    except Exception:
        logger.exception("Failed to record LiveKit LLM step")
