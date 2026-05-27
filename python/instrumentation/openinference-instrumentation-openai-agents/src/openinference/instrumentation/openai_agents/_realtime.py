"""Event-driven instrumentation for openai-agents RealtimeSession.

Wraps RealtimeSession._put_event to observe all RealtimeSessionEvents and
produce OpenInference-compliant AUDIO/USER/LLM/TOOL spans per turn.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import wave
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, Optional
from weakref import WeakKeyDictionary

from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY, get_value, set_value
from opentelemetry.trace import Span, Status, StatusCode, set_span_in_context

from openinference.instrumentation import OITracer, TraceConfig
from openinference.semconv.trace import (
    OpenInferenceLLMSystemValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Instrumentor-local constants (not yet in semconv)
_END_REASON = "end_reason"
_TIME_TO_FIRST_TOKEN_MS = "time_to_first_token_ms"
_END_REASON_COMPLETE = "complete"
_END_REASON_INTERRUPTED = "interrupted"
_END_REASON_SESSION_CLOSED = "session_closed"
_INPUT_AUDIO_URL = "input.audio.url"
_INPUT_AUDIO_MIME_TYPE = "input.audio.mime_type"
_INPUT_AUDIO_TRANSCRIPT = "input.audio.transcript"
_OUTPUT_AUDIO_URL = "output.audio.url"
_OUTPUT_AUDIO_MIME_TYPE = "output.audio.mime_type"
_OUTPUT_AUDIO_TRANSCRIPT = "output.audio.transcript"
_AUDIO_KIND = "AUDIO"
_USER_KIND = "USER"

# Short-hands for frequently used attribute keys
_OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
_INPUT_VALUE = SpanAttributes.INPUT_VALUE
_OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
_INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
_SESSION_ID = SpanAttributes.SESSION_ID
_LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
_LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
_LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
_LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
_LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
_LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
_LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO
_LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO
_TOOL_NAME = SpanAttributes.TOOL_NAME

_LLM_KIND = OpenInferenceSpanKindValues.LLM.value
_TOOL_KIND = OpenInferenceSpanKindValues.TOOL.value
_OPENAI_SYSTEM = OpenInferenceLLMSystemValues.OPENAI.value

# Audio redaction env vars (experimental — promote to shared TraceConfig once stable)
_ENV_HIDE_INPUT_AUDIO = "OPENINFERENCE_HIDE_INPUT_AUDIO"
_ENV_HIDE_OUTPUT_AUDIO = "OPENINFERENCE_HIDE_OUTPUT_AUDIO"
_ENV_BASE64_AUDIO_MAX_LENGTH = "OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH"
_DEFAULT_BASE64_AUDIO_MAX_LENGTH = 32_000


def _env_bool(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _hide_input_audio(config: TraceConfig) -> bool:
    return bool(config.hide_inputs) or _env_bool(_ENV_HIDE_INPUT_AUDIO)


def _hide_output_audio(config: TraceConfig) -> bool:
    return bool(config.hide_outputs) or _env_bool(_ENV_HIDE_OUTPUT_AUDIO)


def _base64_audio_max_length() -> int:
    raw = os.environ.get(_ENV_BASE64_AUDIO_MAX_LENGTH)
    if raw is None:
        return _DEFAULT_BASE64_AUDIO_MAX_LENGTH
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_BASE64_AUDIO_MAX_LENGTH


# OpenAI Realtime API streams 24 kHz mono PCM16 in both directions.
_SAMPLE_RATE_HZ = 24_000
_SAMPLE_WIDTH_BYTES = 2
_NUM_CHANNELS = 1


def pcm16_to_wav_data_uri(
    pcm_bytes: bytes,
    sample_rate: int = _SAMPLE_RATE_HZ,
    num_channels: int = _NUM_CHANNELS,
    sample_width: int = _SAMPLE_WIDTH_BYTES,
) -> str:
    """Encode raw PCM16 bytes to a WAV data: URI (audio/wav)."""
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def truncate_audio_data_uri(uri: str, max_length: int) -> str:
    """Truncate the base64 payload of any base64 ``data:`` URI to max_length chars.

    Named for the audio use-case but works on any ``data:<mediatype>;base64,<payload>`` URI.
    Preserves the ``data:<mediatype>;base64,`` prefix; only the payload is truncated. The
    result is intentionally not valid base64.
    """
    if not uri.startswith("data:") or ";base64," not in uri:
        return uri
    prefix, payload = uri.split(";base64,", 1)
    if len(payload) <= max_length:
        return uri
    return f"{prefix};base64,{payload[:max_length]}"


# Realtime event types — imported lazily at patch time; None means not available.
_RealtimeAgentStartEvent: Any = None
_RealtimeAgentEndEvent: Any = None
_RealtimeAudio: Any = None
_RealtimeAudioInterrupted: Any = None
_RealtimeError: Any = None
_RealtimeRawModelEvent: Any = None
_RealtimeToolStart: Any = None
_RealtimeToolEnd: Any = None
_RealtimeInputAudioTimeoutTriggered: Any = None


def _load_realtime_events() -> bool:
    """Import realtime event types from the agents SDK. Returns True on success."""
    global _RealtimeAgentStartEvent
    global _RealtimeAgentEndEvent
    global _RealtimeAudio
    global _RealtimeAudioInterrupted
    global _RealtimeError
    global _RealtimeRawModelEvent
    global _RealtimeToolStart
    global _RealtimeToolEnd
    global _RealtimeInputAudioTimeoutTriggered
    try:
        from agents.realtime.events import (
            RealtimeAgentEndEvent,
            RealtimeAgentStartEvent,
            RealtimeAudio,
            RealtimeAudioInterrupted,
            RealtimeError,
            RealtimeRawModelEvent,
            RealtimeToolEnd,
            RealtimeToolStart,
        )

        _RealtimeAgentStartEvent = RealtimeAgentStartEvent
        _RealtimeAgentEndEvent = RealtimeAgentEndEvent
        _RealtimeAudio = RealtimeAudio
        _RealtimeAudioInterrupted = RealtimeAudioInterrupted
        _RealtimeError = RealtimeError
        _RealtimeRawModelEvent = RealtimeRawModelEvent
        _RealtimeToolStart = RealtimeToolStart
        _RealtimeToolEnd = RealtimeToolEnd
        # Added in a later openai-agents release; absent on older pinned versions.
        try:
            from agents.realtime.events import (  # type: ignore[attr-defined,unused-ignore]
                RealtimeInputAudioTimeoutTriggered,
            )

            _RealtimeInputAudioTimeoutTriggered = RealtimeInputAudioTimeoutTriggered
        except ImportError:
            pass
        return True
    except ImportError:
        logger.debug("agents.realtime.events not available — realtime instrumentation disabled")
        return False


# Per-session state, keyed weakly so we don't keep dead sessions alive.
_session_states: WeakKeyDictionary[Any, "_RealtimeSessionState"] = WeakKeyDictionary()


@dataclass
class _UserInputState:
    """Mutable state for one USER child span."""

    user_span: Span

    # PCM16 accumulation (24 kHz, 1 ch, 2 bytes/sample)
    user_audio_buf: bytearray = field(default_factory=bytearray)
    user_audio_committed: bool = False
    user_transcript: Optional[str] = None
    user_text: Optional[str] = None
    # True for USER states created from a typed text item — they don't own a
    # speech window, so mic audio and transcripts must not route to them.
    text_only: bool = False
    closed: bool = False


@dataclass
class _AssistantResponseState:
    """Mutable state for one LLM child span."""

    response_id: str
    llm_span: Span
    llm_start_ns: int
    # TTFT origin: timestamp of the user query that immediately preceded this response
    # (input_audio_buffer.committed time). Falls back to llm_start_ns if no user query
    # ended for this response (e.g., model-initiated follow-up).
    ttft_start_ns: Optional[int] = None
    first_audio_delta_ns: Optional[int] = None
    asst_audio_buf: bytearray = field(default_factory=bytearray)
    asst_transcript: Optional[str] = None
    # Accumulated transcript deltas — used when .done never fires (interruption).
    asst_transcript_deltas: list[str] = field(default_factory=list)
    model_name: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    tool_spans: Dict[str, Span] = field(default_factory=dict)
    closed: bool = False


@dataclass
class _TurnState:
    """All mutable state for one logical realtime conversation turn."""

    turn_span: Span
    users: list[_UserInputState] = field(default_factory=list)
    responses: Dict[str, _AssistantResponseState] = field(default_factory=dict)
    response_order: list[str] = field(default_factory=list)
    active_user: Optional[_UserInputState] = None

    # Most-recent input_audio_buffer.committed timestamp — TTFT origin for the next
    # response in this turn (perceived user-to-bot latency).
    user_query_ended_ns: Optional[int] = None

    # True after a response.done arrives with a function_call output item and stays
    # True until the follow-up response (without function_call) closes. Defers turn
    # finalization across the tool round-trip so the follow-up routes back here.
    awaiting_tool_followup: bool = False

    # Sticky: once set, never overwritten.
    end_reason: Optional[str] = None
    closed: bool = False


_PRESPEECH_MAX_BYTES = 3 * _SAMPLE_RATE_HZ * _SAMPLE_WIDTH_BYTES  # 3 s of PCM16 mono

# Realtime session.update keys captured into llm.invocation_parameters. Excludes server-internal
# fields (id, object, expires_at, client_secret) so secrets never reach traces.
_SESSION_CONFIG_KEYS = (
    "instructions",
    "voice",
    "modalities",
    "temperature",
    "max_response_output_tokens",
    "turn_detection",
    "input_audio_format",
    "output_audio_format",
    "input_audio_transcription",
    "input_audio_noise_reduction",
)


def _extract_session_config(session: Dict[str, Any]) -> Dict[str, Any]:
    return {k: session[k] for k in _SESSION_CONFIG_KEYS if k in session}


class _RealtimeSessionState:
    """Per-session instrumentation book-keeping."""

    def __init__(self, tracer: OITracer, config: TraceConfig) -> None:
        self._tracer = tracer
        self._config = config
        # Most-recently opened turn. Kept open across multiple assistant responses until the next
        # logical user input starts after assistant output, or until the session closes.
        self._latest_turn: Optional[_TurnState] = None
        # Snapshot of the OTel context at session creation; used as parent for all turn spans
        self._session_ctx = context_api.get_current()
        self._provider_session_id: Optional[str] = None
        # Rolling buffer of audio sent before speech_started fires; capped at 3 s so a
        # long silence before the first utterance doesn't accumulate indefinitely.
        self._prespeech_audio: bytearray = bytearray()
        # Latest session.update config — snapshotted onto each turn's invocation_parameters.
        self._session_config: Dict[str, Any] = {}
        # Session model — captured separately from session.created / session.updated and
        # written to the parent AUDIO turn span as the top-level llm.model_name attribute.
        self._session_model: Optional[str] = None
        # Turn whose tool round-trip hasn't resolved yet. The next response.created
        # routes here even when _latest_turn has advanced to a barge-in turn.
        self._turn_awaiting_followup: Optional[_TurnState] = None
        # Session-level lookup for in-flight TOOL spans by call_id, used so
        # `conversation.item.created` (function_call_output) can find the span
        # without needing a response_id.
        self._tool_spans_by_call_id: Dict[str, Span] = {}
        # Session-level lookup for USER input states by item_id. Used to (a)
        # dedupe duplicate conversation.item.added/.done events for the same
        # text item, and (b) route transcript_completed / audio_committed events
        # to the right USER even when other USER spans have become "active".
        self._user_states_by_item_id: Dict[str, _UserInputState] = {}

    # ------------------------------------------------------------------
    # User-side events
    # ------------------------------------------------------------------

    def on_send_audio(self, audio_bytes: bytes) -> None:
        """Called for every send_audio() call from the application."""
        turn = self._latest_turn
        active_user = turn.active_user if turn and not turn.closed else None
        # Only route to active_user if it owns a speech window. Text-only USERs
        # must NOT accumulate audio — that audio belongs to a future speech_started.
        if active_user and not active_user.text_only and not active_user.user_audio_committed:
            active_user.user_audio_buf.extend(audio_bytes)
        else:
            # Before speech_started (or while a text USER is active): accumulate
            # in a capped rolling buffer.
            self._prespeech_audio.extend(audio_bytes)
            if len(self._prespeech_audio) > _PRESPEECH_MAX_BYTES:
                # Trim oldest bytes to stay within the cap.
                del self._prespeech_audio[: len(self._prespeech_audio) - _PRESPEECH_MAX_BYTES]

    def on_speech_started(self, item_id: Optional[str]) -> None:
        turn = self._turn_for_new_user_input()
        user_ctx = set_span_in_context(turn.turn_span)
        user_span = self._start_span(
            name="user",
            context=user_ctx,
            attributes={_OPENINFERENCE_SPAN_KIND: _USER_KIND},
        )
        user = _UserInputState(user_span=user_span)
        # Flush pre-speech buffer into the turn so the beginning of the utterance is captured.
        if self._prespeech_audio:
            user.user_audio_buf.extend(self._prespeech_audio)
            self._prespeech_audio = bytearray()
        turn.users.append(user)
        turn.active_user = user
        if item_id:
            self._user_states_by_item_id[item_id] = user
        logger.debug("realtime: opened USER span (turn has %d users)", len(turn.users))

    def on_user_audio_append(self, audio_b64: str) -> None:
        active_user = self._active_user
        if active_user and not active_user.text_only and audio_b64:
            try:
                active_user.user_audio_buf.extend(base64.b64decode(audio_b64))
            except Exception:
                pass

    def on_user_audio_committed(self, item_id: Optional[str]) -> None:
        # Route by item_id when possible; fall back to active_user.
        user = self._user_states_by_item_id.get(item_id) if item_id else None
        if user is None:
            user = self._active_user
        if user and not user.closed:
            user.user_audio_committed = True
        turn = self._latest_turn
        if turn and not turn.closed:
            turn.user_query_ended_ns = time.monotonic_ns()

    def on_user_transcript_completed(self, item_id: Optional[str], transcript: str) -> None:
        # Route by item_id when possible; fall back to most recent non-text USER
        # so interleaved text events (which steal `_active_user`) don't capture
        # the audio transcript.
        user = self._user_states_by_item_id.get(item_id) if item_id else None
        if user is None:
            for candidate in (self._active_user, self._latest_user):
                if candidate and not candidate.text_only:
                    user = candidate
                    break
        if user and not user.text_only and not user.closed:
            user.user_transcript = transcript

    def on_user_text_created(self, item_id: Optional[str], text: str) -> None:
        if not text:
            return
        # Dedupe: server emits both `conversation.item.added` AND `.done` for the
        # same client-sent text item (same item.id). Only the first creates a span.
        if item_id and item_id in self._user_states_by_item_id:
            return
        turn = self._turn_for_new_user_input()
        user_ctx = set_span_in_context(turn.turn_span)
        user_span = self._start_span(
            name="user",
            context=user_ctx,
            attributes={_OPENINFERENCE_SPAN_KIND: _USER_KIND},
        )
        user = _UserInputState(user_span=user_span, user_text=text, text_only=True)
        turn.users.append(user)
        turn.active_user = user
        if item_id:
            self._user_states_by_item_id[item_id] = user

    # ------------------------------------------------------------------
    # Response lifecycle
    # ------------------------------------------------------------------

    def on_session_created(self, session_id: Optional[str]) -> None:
        if session_id and not self._provider_session_id:
            self._provider_session_id = session_id

    def on_session_config(self, session: Dict[str, Any]) -> None:
        cfg = _extract_session_config(session)
        if cfg:
            self._session_config.update(cfg)
        model = session.get("model")
        if isinstance(model, str) and model:
            self._session_model = model

    def on_response_created(self, response_id: str, model_name: Optional[str]) -> None:
        if not response_id:
            return
        # Tool-callback follow-ups route to the originating turn, even when a
        # barge-in has already opened a newer "latest" turn.
        turn = self._turn_awaiting_followup
        if not turn or turn.closed:
            turn = self._latest_turn
        if not turn or turn.closed:
            turn = self._start_turn()
        if response_id in turn.responses:
            return
        logger.debug("realtime: opening LLM span response_id=%s model=%s", response_id, model_name)
        llm_ctx = set_span_in_context(turn.turn_span)
        llm_span = self._start_span(
            name="assistant",
            context=llm_ctx,
            attributes={
                _OPENINFERENCE_SPAN_KIND: _LLM_KIND,
                _LLM_SYSTEM: _OPENAI_SYSTEM,
            },
        )
        response = _AssistantResponseState(
            response_id=response_id,
            llm_span=llm_span,
            llm_start_ns=time.monotonic_ns(),
            ttft_start_ns=turn.user_query_ended_ns,
            model_name=model_name,
        )
        turn.responses[response_id] = response
        turn.response_order.append(response_id)

    def on_audio_delta(self, response_id: str, audio_bytes: bytes) -> None:
        response = self._response(response_id)
        if not response or response.closed or not audio_bytes:
            return
        if response.first_audio_delta_ns is None:
            response.first_audio_delta_ns = time.monotonic_ns()
        response.asst_audio_buf.extend(audio_bytes)

    def on_audio_interrupted(self) -> None:
        # The SDK fires RealtimeAudioInterrupted on every new user turn after any
        # assistant response, because its audio tracker keeps a reference to the
        # last assistant audio item even after response.done. Only honor the signal
        # when an actively in-flight response or pending tool follow-up exists —
        # otherwise normal follow-up questions get tagged INTERRUPTED.
        # Check both _latest_turn (typical barge-in target) and the awaiting-followup
        # turn (which may differ if a barge-in already opened a newer turn).
        for candidate in (self._latest_turn, self._turn_awaiting_followup):
            if not candidate or candidate.closed:
                continue
            in_flight = any(not r.closed for r in candidate.responses.values())
            if in_flight or candidate.awaiting_tool_followup:
                _set_end_reason(candidate, _END_REASON_INTERRUPTED)
                return

    def on_asst_transcript_delta(self, response_id: str, delta: str) -> None:
        response = self._response(response_id)
        if response and not response.closed and delta:
            response.asst_transcript_deltas.append(delta)

    def on_asst_transcript_done(self, response_id: str, transcript: str) -> None:
        response = self._response(response_id)
        if response and not response.closed:
            response.asst_transcript = transcript

    def on_function_call_added(self, response_id: str, call_id: str, tool_name: str) -> None:
        """Raw `response.output_item.added` with item.type=function_call.

        Opens a TOOL span as a child of the LLM span for this response. Arguments
        arrive later via `on_function_call_arguments_done`, output via
        `on_function_call_output`.
        """
        if not response_id or not call_id:
            return
        response = self._response(response_id)
        if not response or response.closed:
            return
        if call_id in response.tool_spans:
            return  # idempotent: SDK can re-emit
        tool_ctx = set_span_in_context(response.llm_span)
        attrs: Dict[str, Any] = {_OPENINFERENCE_SPAN_KIND: _TOOL_KIND}
        if tool_name:
            attrs[_TOOL_NAME] = tool_name
        tool_span = self._start_span(
            name=tool_name or "tool",
            context=tool_ctx,
            attributes=attrs,
        )
        response.tool_spans[call_id] = tool_span
        self._tool_spans_by_call_id[call_id] = tool_span

    def on_function_call_arguments_done(
        self, response_id: str, call_id: str, arguments: str
    ) -> None:
        """Raw `response.function_call_arguments.done`. Sets the JSON args on the
        already-open TOOL span (opened earlier by output_item.added)."""
        if not call_id or not arguments:
            return
        tool_span = self._tool_spans_by_call_id.get(call_id)
        if tool_span is None:
            # output_item.added may not have arrived first in some flows — start now.
            self.on_function_call_added(response_id, call_id, tool_name="")
            tool_span = self._tool_spans_by_call_id.get(call_id)
            if tool_span is None:
                return
        tool_span.set_attribute(_INPUT_VALUE, arguments)
        tool_span.set_attribute(_INPUT_MIME_TYPE, "application/json")

    def on_function_call_output(self, call_id: str, output: Optional[str]) -> None:
        """Raw `conversation.item.created` with item.type=function_call_output.

        Sets the tool's return value and ends the TOOL span. The call_id alone
        identifies the span — no response_id needed on this event.
        """
        if not call_id:
            return
        tool_span = self._tool_spans_by_call_id.pop(call_id, None)
        if tool_span is None:
            return
        # Also remove from the per-response registry so finalize-cleanup doesn't
        # try to re-end it.
        for turn in (self._latest_turn, self._turn_awaiting_followup):
            if turn is None:
                continue
            for response in turn.responses.values():
                response.tool_spans.pop(call_id, None)
        if output is not None:
            tool_span.set_attribute(_OUTPUT_VALUE, output)
        tool_span.set_status(Status(StatusCode.OK))
        tool_span.end()

    def on_response_done(
        self,
        response_id: str,
        usage: Optional[Dict[str, Any]],
        model_name: Optional[str],
        output: Optional[list[Any]] = None,
    ) -> None:
        response = self._response(response_id)
        if not response or response.closed:
            logger.debug(
                "realtime: on_response_done response_id=%s — no open response to finalize",
                response_id,
            )
            return
        items = output or []
        # Did this response declare a tool call? If so, the parent turn must stay open
        # for the follow-up response.
        has_function_call = any(
            isinstance(item, dict) and item.get("type") == "function_call" for item in items
        )
        turn = self._turn_for_response(response_id)
        logger.debug(
            "realtime: finalizing LLM span response_id=%s function_call=%s",
            response_id,
            has_function_call,
        )
        _finalize_response(response, self._config, usage=usage, model_name=model_name)

        if not turn or turn.closed:
            return
        if has_function_call:
            turn.awaiting_tool_followup = True
            self._turn_awaiting_followup = turn
        else:
            # Final follow-up landed — clear awaiting state and finalize if the user
            # already moved on (turn is no longer latest).
            turn.awaiting_tool_followup = False
            if self._turn_awaiting_followup is turn:
                self._turn_awaiting_followup = None
            if turn is not self._latest_turn:
                _finalize_turn(turn, self._config)

    def on_error(self, error_message: str) -> None:
        # RealtimeError has no response_id field, so errors land on the latest turn.
        turn = self._latest_turn
        if not turn or turn.closed:
            return
        _set_end_reason(turn, _END_REASON_SESSION_CLOSED)
        status = Status(StatusCode.ERROR, description=error_message)
        _finalize_turn(turn, self._config, status=status)

    def on_session_close(self) -> None:
        if self._latest_turn and not self._latest_turn.closed:
            responses = self._latest_turn.responses.values()
            if not self._latest_turn.responses or any(
                not response.closed for response in responses
            ):
                _set_end_reason(self._latest_turn, _END_REASON_SESSION_CLOSED)
            logger.debug(
                "realtime: on_session_close — finalizing turn with %d user(s), %d response(s)",
                len(self._latest_turn.users),
                len(self._latest_turn.responses),
            )
            _finalize_turn(self._latest_turn, self._config)
        else:
            logger.debug(
                "realtime: on_session_close — no open turn to finalize (latest_turn=%s)",
                "closed" if self._latest_turn else "None",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _active_user(self) -> Optional[_UserInputState]:
        turn = self._latest_turn
        return turn.active_user if turn and not turn.closed else None

    @property
    def _latest_user(self) -> Optional[_UserInputState]:
        turn = self._latest_turn
        if not turn or turn.closed or not turn.users:
            return None
        return turn.users[-1]

    def _start_turn(self) -> _TurnState:
        attributes: Dict[str, Any] = {_OPENINFERENCE_SPAN_KIND: _AUDIO_KIND}
        if self._session_model:
            attributes[_LLM_MODEL_NAME] = self._session_model
        if self._session_config:
            try:
                attributes[_LLM_INVOCATION_PARAMETERS] = json.dumps(self._session_config)
            except (TypeError, ValueError):
                pass
        turn_span = self._start_span(
            name="conversation.turn",
            context=self._session_ctx,
            attributes=attributes,
        )
        turn = _TurnState(turn_span=turn_span)
        self._latest_turn = turn
        logger.debug("realtime: opened AUDIO turn span")
        return turn

    def _turn_for_new_user_input(self) -> _TurnState:
        turn = self._latest_turn
        if not turn or turn.closed:
            return self._start_turn()
        if turn.awaiting_tool_followup:
            # Mid-tool-round-trip barge-in: the originating turn must stay open so
            # the follow-up response can attach to it. Mark INTERRUPTED (sticky)
            # and open a new turn for the new user input.
            _set_end_reason(turn, _END_REASON_INTERRUPTED)
            return self._start_turn()
        if turn.responses:
            if any(not response.closed for response in turn.responses.values()):
                _set_end_reason(turn, _END_REASON_INTERRUPTED)
            _finalize_turn(turn, self._config)
            return self._start_turn()
        return turn

    def _turn_for_response(self, response_id: Optional[str]) -> Optional[_TurnState]:
        if not response_id:
            return None
        # The response may belong to a turn that's no longer _latest_turn — e.g., a
        # tool-call follow-up whose originating turn deferred finalization while a
        # barge-in opened the latest turn.
        for turn in (self._latest_turn, self._turn_awaiting_followup):
            if turn and not turn.closed and response_id in turn.responses:
                return turn
        return None

    def _response(self, response_id: str) -> Optional[_AssistantResponseState]:
        turn = self._turn_for_response(response_id)
        return turn.responses.get(response_id) if turn else None

    def _start_span(self, name: str, context: Any, attributes: Dict[str, Any]) -> Span:
        ctx = self._session_ctx
        if self._provider_session_id and get_value(_SESSION_ID, ctx) is None:
            ctx = set_value(_SESSION_ID, self._provider_session_id, ctx)
        token = context_api.attach(ctx)
        try:
            return self._tracer.start_span(
                name=name,
                context=context,
                attributes=attributes,
            )
        finally:
            context_api.detach(token)


# ------------------------------------------------------------------
# Turn finalization (shared by on_response_done and on_session_close)
# ------------------------------------------------------------------


def _finalize_turn(
    turn: _TurnState,
    config: TraceConfig,
    status: Optional[Status] = None,
) -> None:
    if turn.closed:
        logger.debug("realtime: _finalize_turn — turn already closed, skipping")
        return
    logger.debug(
        "realtime: _finalize_turn — ending AUDIO + %d USER + %d LLM spans",
        len(turn.users),
        len(turn.responses),
    )
    status = status or Status(StatusCode.OK)

    for response_id in turn.response_order:
        response = turn.responses.get(response_id)
        if response:
            _finalize_response(response, config, status=status)

    # Close any tool spans that never received their function_call_output
    # (session close / error mid-tool-execution). Tool spans intentionally
    # outlive their LLM parent — they get cleaned up here at turn finalize.
    # By the time these reach cleanup the tool never returned, so always mark
    # ERROR regardless of the turn's status.
    orphan_status = Status(StatusCode.ERROR, description="tool did not return before turn ended")
    for response in turn.responses.values():
        for ts in response.tool_spans.values():
            ts.set_status(orphan_status)
            ts.end()
        response.tool_spans.clear()

    for user in turn.users:
        _finalize_user(user, config, status=status)

    # --- AUDIO turn parent ---
    _set_end_reason(turn, _END_REASON_COMPLETE)
    turn_span = turn.turn_span
    _set_turn_io_attributes(turn, config)
    turn_span.set_attribute(_END_REASON, turn.end_reason or _END_REASON_COMPLETE)
    turn_span.set_status(status)
    turn_span.end()

    turn.closed = True


def _finalize_response(
    response: _AssistantResponseState,
    config: TraceConfig,
    usage: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    status: Optional[Status] = None,
) -> None:
    if response.closed:
        return
    # On interruption, response.output_audio_transcript.done never fires — fall back to
    # the accumulated deltas so partial transcripts survive on the LLM and parent spans.
    if not response.asst_transcript and response.asst_transcript_deltas:
        response.asst_transcript = "".join(response.asst_transcript_deltas)
    hide_out = _hide_output_audio(config)
    max_len = _base64_audio_max_length()
    status = status or Status(StatusCode.OK)
    llm_span = response.llm_span
    effective_model = model_name or response.model_name
    effective_usage = usage or response.usage
    if model_name:
        response.model_name = model_name
    if usage:
        response.usage = usage

    if effective_model:
        llm_span.set_attribute(_LLM_MODEL_NAME, effective_model)

    if effective_usage:
        if (pt := effective_usage.get("input_tokens")) is not None:
            llm_span.set_attribute(_LLM_TOKEN_COUNT_PROMPT, pt)
        if (ct := effective_usage.get("output_tokens")) is not None:
            llm_span.set_attribute(_LLM_TOKEN_COUNT_COMPLETION, ct)
        if (tt := effective_usage.get("total_tokens")) is not None:
            llm_span.set_attribute(_LLM_TOKEN_COUNT_TOTAL, tt)
        if id_ := effective_usage.get("input_token_details"):
            if (at := id_.get("audio_tokens")) is not None:
                llm_span.set_attribute(_LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO, at)
        if od := effective_usage.get("output_token_details"):
            if (at := od.get("audio_tokens")) is not None:
                llm_span.set_attribute(_LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO, at)

    if response.first_audio_delta_ns is not None:
        ttft_origin_ns = (
            response.ttft_start_ns if response.ttft_start_ns is not None else response.llm_start_ns
        )
        # Clamp to 0 — in tight timings (audio delta arriving before user_query_ended_ns
        # is captured) the subtraction can go negative; a negative TTFT is meaningless.
        ttft_ms = max(0, int((response.first_audio_delta_ns - ttft_origin_ns) / 1_000_000))
        llm_span.set_attribute(_TIME_TO_FIRST_TOKEN_MS, ttft_ms)

    if not hide_out and response.asst_audio_buf:
        uri = pcm16_to_wav_data_uri(bytes(response.asst_audio_buf))
        if len(uri) > max_len:
            # spec/audio_spans.md: preserve the data:<mediatype>;base64, prefix
            uri = truncate_audio_data_uri(uri, max_len)
        llm_span.set_attribute(_OUTPUT_AUDIO_URL, uri)
        llm_span.set_attribute(_OUTPUT_AUDIO_MIME_TYPE, "audio/wav")

    if not hide_out and response.asst_transcript:
        llm_span.set_attribute(_OUTPUT_AUDIO_TRANSCRIPT, response.asst_transcript)

    # NOTE: Tool spans intentionally outlive their LLM parent. response.done fires
    # when the server is done with the assistant's pre-tool reasoning; the tool
    # then executes and emits its own conversation.item.created (function_call_output)
    # later, which is what ends the TOOL span. Any tool spans still open at turn
    # finalization (session close, error) are cleaned up there.

    llm_span.set_status(status)
    llm_span.end()
    response.closed = True


def _finalize_user(
    user: _UserInputState,
    config: TraceConfig,
    status: Optional[Status] = None,
) -> None:
    if user.closed:
        return
    hide_audio = _hide_input_audio(config)
    hide_text = config.hide_inputs or config.hide_input_text
    max_len = _base64_audio_max_length()
    status = status or Status(StatusCode.OK)
    user_span = user.user_span
    if not hide_audio and user.user_audio_buf:
        uri = pcm16_to_wav_data_uri(bytes(user.user_audio_buf))
        if len(uri) > max_len:
            uri = truncate_audio_data_uri(uri, max_len)
        user_span.set_attribute(_INPUT_AUDIO_URL, uri)
        user_span.set_attribute(_INPUT_AUDIO_MIME_TYPE, "audio/wav")
    if not hide_audio and user.user_transcript:
        user_span.set_attribute(_INPUT_AUDIO_TRANSCRIPT, user.user_transcript)
    if not hide_text and user.user_text:
        user_span.set_attribute(_INPUT_VALUE, user.user_text)
        user_span.set_attribute(_INPUT_MIME_TYPE, "text/plain")
    user_span.set_status(status)
    user_span.end()
    user.closed = True


def _set_turn_io_attributes(turn: _TurnState, config: TraceConfig) -> None:
    # Mirror child-span gating: the parent AUDIO span aggregates user/assistant
    # text from the same source fields the children read, so the same flags must
    # suppress it here too. Without this, hide_inputs/hide_outputs (and the audio
    # env vars) leak transcripts onto the parent even when the child is clean.
    if not (config.hide_inputs or _hide_input_audio(config)):
        input_values = [
            text for user in turn.users for text in (user.user_text, user.user_transcript) if text
        ]
        if input_values:
            turn.turn_span.set_attribute(_INPUT_VALUE, "\n".join(input_values))
    if not (config.hide_outputs or _hide_output_audio(config)):
        output_values = [
            response.asst_transcript
            for response_id in turn.response_order
            if (response := turn.responses.get(response_id)) and response.asst_transcript
        ]
        if output_values:
            turn.turn_span.set_attribute(_OUTPUT_VALUE, "\n".join(output_values))


def _set_end_reason(turn: _TurnState, reason: str) -> None:
    if not turn.end_reason:  # sticky
        turn.end_reason = reason


# ------------------------------------------------------------------
# Event dispatch
# ------------------------------------------------------------------


def _dispatch_event(state: _RealtimeSessionState, event: Any) -> None:
    if _RealtimeAgentEndEvent is not None and isinstance(event, _RealtimeAgentEndEvent):
        # An agent's logical run ended (often due to interruption). This does NOT
        # mean the session is over — there may already be a new turn open for a
        # barge-in user input. Per-response finalization happens via response.done,
        # so this event needs no action here. Closing the latest turn here was a
        # bug: it produced phantom empty AUDIO spans when a user cut in mid-playback.
        logger.debug("realtime: RealtimeAgentEndEvent — no-op")
        return

    if _RealtimeInputAudioTimeoutTriggered is not None and isinstance(
        event, _RealtimeInputAudioTimeoutTriggered
    ):
        state.on_session_close()
        return

    if _RealtimeAudio is not None and isinstance(event, _RealtimeAudio):
        # event.audio is RealtimeModelAudioEvent; bytes are at .data, response_id at .response_id
        audio_event = getattr(event, "audio", None)
        audio_bytes: bytes = getattr(audio_event, "data", b"") or b""
        response_id = str(getattr(audio_event, "response_id", None) or "")
        if not response_id:
            logger.debug(
                "realtime: RealtimeAudio with no response_id — dropping %d bytes",
                len(audio_bytes),
            )
            return
        state.on_audio_delta(response_id, audio_bytes)
        return

    if _RealtimeAudioInterrupted is not None and isinstance(event, _RealtimeAudioInterrupted):
        state.on_audio_interrupted()
        return

    # SDK RealtimeToolStart/RealtimeToolEnd events don't carry response_id or
    # call_id, so we use the raw events instead (see _dispatch_raw below).

    if _RealtimeError is not None and isinstance(event, _RealtimeError):
        err = getattr(event, "error", None)
        state.on_error(str(err) if err else "realtime error")
        return

    if _RealtimeRawModelEvent is not None and isinstance(event, _RealtimeRawModelEvent):
        _dispatch_raw(state, event)


def _dispatch_raw(state: _RealtimeSessionState, event: Any) -> None:
    # RealtimeRawModelEvent.data is a RealtimeModelEvent subclass.
    # For raw server events it is RealtimeModelRawServerEvent whose .data is the server dict.
    inner = getattr(event, "data", None)
    raw: Any = getattr(inner, "data", None)
    if not isinstance(raw, dict):
        return
    etype: str = raw.get("type") or ""
    logger.debug("realtime raw: %s", etype)

    if etype == "input_audio_buffer.speech_started":
        state.on_speech_started(raw.get("item_id"))

    elif etype == "session.created":
        session = raw.get("session") or {}
        state.on_session_created(session.get("id"))
        state.on_session_config(session)

    elif etype == "session.updated":
        session = raw.get("session") or {}
        state.on_session_config(session)

    elif etype == "input_audio_buffer.append":
        state.on_user_audio_append(raw.get("audio") or "")

    elif etype == "input_audio_buffer.committed":
        state.on_user_audio_committed(raw.get("item_id"))

    elif etype == "conversation.item.input_audio_transcription.completed":
        state.on_user_transcript_completed(raw.get("item_id"), raw.get("transcript") or "")

    elif etype == "response.created":
        resp: Dict[str, Any] = raw.get("response") or {}
        state.on_response_created(resp.get("id") or "", resp.get("model"))

    elif etype == "response.output_audio_transcript.delta":
        state.on_asst_transcript_delta(raw.get("response_id") or "", raw.get("delta") or "")

    elif etype == "response.output_audio_transcript.done":
        state.on_asst_transcript_done(raw.get("response_id") or "", raw.get("transcript") or "")

    elif etype == "response.output_item.added":
        item = raw.get("item") or {}
        if item.get("type") == "function_call":
            # Key tool spans by call_id (linkage id used by function_call_output),
            # NOT item.id (which is the conversation-item id; the two are distinct
            # in the GA API).
            state.on_function_call_added(
                raw.get("response_id") or "",
                item.get("call_id") or item.get("id") or "",
                item.get("name") or "",
            )

    elif etype == "response.function_call_arguments.done":
        # Tool spans are keyed by call_id; item_id is a different identifier and
        # using it as a fallback would silently miss the actual tool span.
        state.on_function_call_arguments_done(
            raw.get("response_id") or "",
            raw.get("call_id") or "",
            raw.get("arguments") or "",
        )

    elif etype in (
        "conversation.item.created",
        "conversation.item.added",
        "conversation.item.done",
    ):
        # Pre-GA: `.created`. GA: `.added` when item first appears, `.done` when
        # finalized with full content. For client-sent function_call_output items,
        # observed behavior varies between API versions, so handle all three.
        item = raw.get("item") or {}
        if item.get("type") == "function_call_output":
            output = item.get("output")
            if output is not None:
                state.on_function_call_output(
                    item.get("call_id") or "",
                    str(output),
                )
        else:
            text = _extract_user_text(item)
            if text:
                state.on_user_text_created(item.get("id") or "", text)

    elif etype == "response.done":
        resp = raw.get("response") or {}
        state.on_response_done(
            resp.get("id") or "",
            resp.get("usage"),
            resp.get("model"),
            resp.get("output"),
        )


def _extract_user_text(item: Dict[str, Any]) -> str:
    if item.get("role") != "user":
        return ""
    content = item.get("content") or []
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in ("input_text", "text"):
            text = part.get("text")
            if text:
                parts.append(str(text))
    return "\n".join(parts)


# ------------------------------------------------------------------
# Patch factory
# ------------------------------------------------------------------


def make_realtime_wrapper(tracer: OITracer, config: TraceConfig) -> Any:
    """Return an async wrapt wrapper for RealtimeSession._put_event."""

    async def _put_event_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        # SDK always calls `self._put_event(event)` positionally.
        if not args:
            return await wrapped(*args, **kwargs)
        event = args[0]

        if instance not in _session_states:
            _session_states[instance] = _RealtimeSessionState(tracer, config)
            logger.debug(
                "realtime: created session state for %s id=%s",
                type(instance).__name__,
                id(instance),
            )
        state = _session_states[instance]

        logger.debug("realtime _put_event: %s", type(event).__name__)
        try:
            _dispatch_event(state, event)
        except Exception:
            logger.debug("Exception in realtime event dispatch", exc_info=True)

        return await wrapped(*args, **kwargs)

    return _put_event_wrapper


def make_send_audio_wrapper() -> Any:
    """Return an async wrapt wrapper for RealtimeSession.send_audio."""

    async def _send_audio_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        audio = args[0] if args else kwargs.get("audio", b"")

        if instance in _session_states and isinstance(audio, (bytes, bytearray)):
            try:
                _session_states[instance].on_send_audio(bytes(audio))
            except Exception:
                logger.debug("Exception in send_audio instrumentation", exc_info=True)

        return await wrapped(*args, **kwargs)

    return _send_audio_wrapper


def make_close_wrapper() -> Any:
    """Return an async wrapt wrapper for RealtimeSession.close.

    Finalizes any open turn so AUDIO/USER spans are ended on clean shutdown
    (including `async with` exit and explicit close()).
    """

    async def _close_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        state = _session_states.get(instance)
        logger.debug(
            "realtime close(): instance=%s state=%s",
            type(instance).__name__,
            "present" if state is not None else "missing",
        )
        if state is not None:
            try:
                state.on_session_close()
            except Exception:
                logger.debug("Exception in realtime close instrumentation", exc_info=True)
        return await wrapped(*args, **kwargs)

    return _close_wrapper
