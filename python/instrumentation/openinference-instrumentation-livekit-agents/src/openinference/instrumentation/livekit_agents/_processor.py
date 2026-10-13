from __future__ import annotations

import logging
import threading
from numbers import Real
from typing import Any, Literal, Mapping, cast

from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY, get_value
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import Span, StatusCode

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.livekit_agents._fold import (
    AgentTurnEnded,
    AgentTurnStarted,
    AudioUsageNoted,
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
    TraceEvent,
    UserTurnEnded,
    VoiceTurn,
    reduce_trace,
)
from openinference.instrumentation.livekit_agents._render import render_turn
from openinference.semconv.trace import OpenInferenceLLMProviderValues, SpanAttributes

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_SCOPE = "livekit-agents"
_Phase = Literal["start", "end"]

_CONVERSATION_ID = "gen_ai.conversation.id"
_ROOM_SID = "lk.room_sid"
_SPEECH_ID = "lk.speech_id"
_AGENT_NAME = "gen_ai.agent.name"
_USER_TRANSCRIPT = "lk.pii.user_transcript"
_USER_INPUT = "lk.pii.user_input"
_TRANSCRIPT_CONFIDENCE = "lk.transcript_confidence"
_REQUEST_MODEL = "gen_ai.request.model"
_PROVIDER_NAME = "gen_ai.provider.name"
_INPUT_MESSAGES = "gen_ai.input.messages"
_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
_OUTPUT_MESSAGES = "gen_ai.output.messages"
_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_AUDIO_INPUT_TOKENS = "gen_ai.usage.audio.input_tokens"
_AUDIO_OUTPUT_TOKENS = "gen_ai.usage.audio.output_tokens"
_TTFT_SECONDS = "gen_ai.response.time_to_first_chunk"
_NODE_TTFT = "lk.response.ttft"
_TTS_TEXT = "lk.pii.input_text"
_INTERRUPTED = "lk.interrupted"
_INTERRUPTION_SOURCE = "lk.interruption.source"
_TOOL_NAME = "gen_ai.tool.name"
_TOOL_NAME_LK = "lk.function_tool.name"
_TOOL_CALL_ID = "gen_ai.tool.call.id"
_TOOL_CALL_ID_LK = "lk.function_tool.id"
_TOOL_ARGS = "gen_ai.tool.call.arguments"
_TOOL_ARGS_LK = "lk.pii.function_tool.arguments"
_TOOL_RESULT = "gen_ai.tool.call.result"
_TOOL_RESULT_LK = "lk.pii.function_tool.output"
_TOOL_IS_ERROR = "lk.function_tool.is_error"


def room_session_id(attributes: Mapping[str, Any] | None) -> str | None:
    if not attributes:
        return None
    for key in (_CONVERSATION_ID, _ROOM_SID):
        value = attributes.get(key)
        if isinstance(value, str) and value:
            return value
    return None


class LiveKitTurnProcessor(SpanProcessor):
    def __init__(self, tracer: OITracer, config: TraceConfig) -> None:
        self._tracer = tracer
        self._config = config
        self._lock = threading.Lock()
        self._folds: dict[int, SessionFold] = {}
        self._enabled = True

    def on_start(self, span: Span, parent_context: context_api.Context | None = None) -> None:
        try:
            if (
                not self._enabled
                or get_value(_SUPPRESS_INSTRUMENTATION_KEY, parent_context)
                or not _has_livekit_scope(span)
            ):
                return
            attributes = getattr(span, "attributes", None)
            if (
                hasattr(span, "set_attribute")
                and (not attributes or SpanAttributes.SESSION_ID not in attributes)
                and (session_id := room_session_id(attributes)) is not None
            ):
                span.set_attribute(SpanAttributes.SESSION_ID, session_id)
            event = parse(cast(ReadableSpan, span), "start")
            if event is not None:
                self._reduce(event)
        except Exception:
            logger.exception("Failed to process LiveKit span start")

    def on_end(self, span: ReadableSpan) -> None:
        try:
            if (
                not self._enabled
                or get_value(_SUPPRESS_INSTRUMENTATION_KEY)
                or not _has_livekit_scope(span)
            ):
                return
            event = parse(span, "end")
            if event is None:
                return
            turns = self._reduce(event)
            for turn in turns:
                render_turn(turn, self._tracer, self._config)
        except Exception:
            logger.exception("Failed to process LiveKit span end")

    def shutdown(self) -> None:
        try:
            with self._lock:
                folds = self._folds
                self._folds = {}
                turns: list[VoiceTurn] = []
                for trace_id, fold in folds.items():
                    span_id = next(
                        (span_id for span_id in fold.open_turns if span_id not in fold.ended),
                        None,
                    )
                    if span_id is None:
                        span_id = next(
                            (span_id for span_id in fold.spans if span_id not in fold.ended),
                            None,
                        )
                    if span_id is None:
                        continue
                    parent_id = fold.spans[span_id].parent_id
                    _, sealed = reduce_trace(
                        fold,
                        SessionEnded(
                            trace_id=trace_id,
                            span_id=span_id,
                            parent_id=parent_id,
                            name="agent_session",
                            phase="end",
                        ),
                    )
                    turns.extend(sealed)
            for turn in turns:
                render_turn(turn, self._tracer, self._config)
        except Exception:
            logger.exception("Failed to shut down LiveKit turn processor")

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def _reduce(self, event: TraceEvent) -> tuple[VoiceTurn, ...]:
        with self._lock:
            state, turns = reduce_trace(self._folds.get(event.trace_id, SessionFold()), event)
            if state.closed:
                self._folds.pop(event.trace_id, None)
            else:
                self._folds[event.trace_id] = state
        return turns


def parse(span: ReadableSpan, phase: _Phase) -> TraceEvent | None:
    if not _has_livekit_scope(span) or span.context is None:
        return None

    context = span.context
    parent_id = span.parent.span_id if span.parent is not None else None
    start_ns = span.start_time or 0
    end_ns = span.end_time if span.end_time is not None else start_ns
    attributes = span.attributes or {}
    identity = {
        "trace_id": context.trace_id,
        "span_id": context.span_id,
        "parent_id": parent_id,
        "name": span.name,
        "phase": phase,
    }

    if span.name == "user_turn" and phase == "end":
        transcript = _nonempty_string(attributes.get(_USER_TRANSCRIPT))
        utterance = None
        if transcript is not None:
            utterance = HeardUtterance(
                transcript=transcript,
                stt_model=_string(attributes.get(_REQUEST_MODEL)),
                stt_provider=_provider(attributes.get(_PROVIDER_NAME)),
                confidence=_number(attributes.get(_TRANSCRIPT_CONFIDENCE)),
                start_ns=start_ns,
                end_ns=end_ns,
            )
        return UserTurnEnded(**identity, utterance=utterance)

    if span.name == "agent_turn":
        if phase == "start":
            return AgentTurnStarted(
                **identity,
                start_ns=start_ns,
                speech_id=_string(attributes.get(_SPEECH_ID)),
                session_id=room_session_id(attributes),
                agent_name=_string(attributes.get(_AGENT_NAME)),
            )
        interruption_source = _nonempty_string(attributes.get(_INTERRUPTION_SOURCE))
        status = getattr(span, "status", None)
        return AgentTurnEnded(
            **identity,
            end_ns=end_ns,
            session_id=room_session_id(attributes),
            speech_id=_string(attributes.get(_SPEECH_ID)),
            agent_name=_string(attributes.get(_AGENT_NAME)),
            typed_text=_nonempty_string(attributes.get(_USER_INPUT)),
            interrupted=attributes.get(_INTERRUPTED) is True or interruption_source is not None,
            failed=status is not None and status.status_code is StatusCode.ERROR,
        )

    if span.name in ("llm_request", "realtime_inference"):
        if phase == "start":
            return SpanIndexed(**identity)
        return ModelStepEnded(
            **identity,
            step=ModelStep(
                span_id=context.span_id,
                source=cast(Literal["llm_request", "realtime_inference"], span.name),
                model=_string(attributes.get(_REQUEST_MODEL)),
                provider=_provider(attributes.get(_PROVIDER_NAME)),
                input_messages_json=_string(attributes.get(_INPUT_MESSAGES)),
                output_messages_json=_string(attributes.get(_OUTPUT_MESSAGES)),
                spoken_text=None,
                input_tokens=_integer(attributes.get(_INPUT_TOKENS)),
                output_tokens=_integer(attributes.get(_OUTPUT_TOKENS)),
                audio_input_tokens=_integer(attributes.get(_AUDIO_INPUT_TOKENS)),
                audio_output_tokens=_integer(attributes.get(_AUDIO_OUTPUT_TOKENS)),
                time_to_first_token_s=_nonnegative_number(attributes.get(_TTFT_SECONDS)),
                start_ns=start_ns,
                end_ns=end_ns,
                system_instructions_json=_string(attributes.get(_SYSTEM_INSTRUCTIONS)),
            ),
        )

    if span.name == "function_tool" and phase == "end":
        return ToolEnded(
            **identity,
            tool=ToolCall(
                name=_preferred_string(attributes, _TOOL_NAME, _TOOL_NAME_LK) or "",
                call_id=_preferred_string(attributes, _TOOL_CALL_ID, _TOOL_CALL_ID_LK),
                arguments=_preferred_string(attributes, _TOOL_ARGS, _TOOL_ARGS_LK),
                output=_preferred_string(attributes, _TOOL_RESULT, _TOOL_RESULT_LK),
                is_error=attributes.get(_TOOL_IS_ERROR) is True,
                start_ns=start_ns,
                end_ns=end_ns,
                step_span_id=None,
            ),
        )

    if span.name == "tts_request" and phase == "end":
        if (text := _nonempty_string(attributes.get(_TTS_TEXT))) is not None:
            return SpokenTextEnded(**identity, text=text, start_ns=start_ns)
        return SpanIndexed(**identity)

    if span.name == "llm_node" and phase == "end":
        if (latency := _nonnegative_number(attributes.get(_NODE_TTFT))) is not None:
            return StepLatencyNoted(**identity, time_to_first_token_s=latency)
        return SpanIndexed(**identity)

    if span.name == "realtime_metrics" and phase == "end":
        return AudioUsageNoted(
            **identity,
            audio_input_tokens=_integer(attributes.get(_AUDIO_INPUT_TOKENS)),
            audio_output_tokens=_integer(attributes.get(_AUDIO_OUTPUT_TOKENS)),
            time_to_first_token_s=_nonnegative_number(attributes.get(_TTFT_SECONDS)),
        )

    if span.name == "agent_session" and phase == "end":
        return SessionEnded(**identity)

    return SpanIndexed(**identity)


# Some LiveKit plugins report the API host as gen_ai.provider.name.
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


def _provider(value: Any) -> str | None:
    provider = _nonempty_string(value)
    if provider is None:
        return None
    return _PROVIDER_HOSTS.get(provider.lower(), provider)


def _has_livekit_scope(span: Any) -> bool:
    scope = getattr(span, "instrumentation_scope", None)
    return scope is not None and scope.name == _SCOPE


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _nonempty_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, Real) and not isinstance(value, bool) else None


def _nonnegative_number(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number >= 0 else None


def _preferred_string(attributes: Mapping[str, Any], primary: str, fallback: str) -> str | None:
    return _string(attributes.get(primary)) or _string(attributes.get(fallback))
