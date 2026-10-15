from __future__ import annotations

import logging
import os
from typing import Any

from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY, get_value
from opentelemetry.trace import Span, Status, StatusCode, set_span_in_context

from openinference.instrumentation import (
    Message,
    OITracer,
    ToolCall,
    TraceConfig,
    get_attributes_from_context,
    get_llm_input_message_attributes,
    get_llm_output_message_attributes,
    using_session,
)
from openinference.instrumentation.livekit_agents._audio import SessionAudio
from openinference.instrumentation.livekit_agents._messages import messages_json
from openinference.instrumentation.livekit_agents._turns import (
    ModelStep,
    ToolRun,
    Utterance,
    VoiceTurn,
)
from openinference.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
_END_REASON = "end_reason"
_TIME_TO_FIRST_TOKEN_MS = "time_to_first_token_ms"
_INPUT_AUDIO_TRANSCRIPT = "input.audio.transcript"
_OUTPUT_AUDIO_TRANSCRIPT = "output.audio.transcript"
_INPUT_AUDIO_URL = "input.audio.url"
_INPUT_AUDIO_MIME_TYPE = "input.audio.mime_type"
_OUTPUT_AUDIO_URL = "output.audio.url"
_OUTPUT_AUDIO_MIME_TYPE = "output.audio.mime_type"
_WAV = "audio/wav"
_TEXT = "text/plain"
_JSON = "application/json"
_ENV_BASE64_AUDIO_MAX_LENGTH = "OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH"
# About 62 seconds of 24 kHz mono PCM16. Arize AX ingests and stores clips of this size.
_DEFAULT_BASE64_AUDIO_MAX_LENGTH = 4_000_000


def render_turn(
    turn: VoiceTurn,
    tracer: OITracer,
    config: TraceConfig,
    audio: SessionAudio | None = None,
) -> None:
    if get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return
    caller_session = dict(get_attributes_from_context()).get(SpanAttributes.SESSION_ID)
    if caller_session is None and turn.session_id is not None:
        with using_session(turn.session_id):
            _render(turn, tracer, config, audio)
    else:
        _render(turn, tracer, config, audio)


def _render(
    turn: VoiceTurn, tracer: OITracer, config: TraceConfig, audio: SessionAudio | None
) -> None:
    turn_span: Span | None = None
    try:
        # A turn is a root span: it is rendered when the speech ends, outside any caller span.
        turn_span = tracer.start_span(
            "conversation.turn",
            context=context_api.Context(),
            attributes={_SPAN_KIND: "AUDIO"},
            start_time=turn.start_ns,
        )
        _set_turn_attributes(turn_span, turn)
        if audio is not None:
            _set_turn_audio(turn_span, turn, audio, config)
        parent = set_span_in_context(turn_span, context_api.Context())
        if turn.user is not None:
            _render_user(turn.user, tracer, parent, config)
        for step in turn.steps:
            _render_step(turn, step, tracer, parent, config)
        for tool in turn.tools:
            if turn.step_for(tool) is None:
                _render_tool(tool, tracer, parent)
    except Exception:
        logger.exception("Failed to render LiveKit conversation turn")
    finally:
        if turn_span is not None:
            _end(turn_span, turn.end_ns, failed=False)


def _set_turn_attributes(span: Span, turn: VoiceTurn) -> None:
    span.set_attribute(_END_REASON, turn.end_reason)
    if (model := next((step.model for step in turn.steps if step.model), None)) is not None:
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model)
    if turn.user is not None:
        span.set_attribute(SpanAttributes.INPUT_VALUE, turn.user.transcript)
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, _TEXT)
    if turn.spoken_text is not None:
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, turn.spoken_text)
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, _TEXT)


def _set_turn_audio(span: Span, turn: VoiceTurn, audio: SessionAudio, config: TraceConfig) -> None:
    max_length = _base64_audio_max_length()
    if turn.user is not None and not hide_input_audio(config):
        heard = audio.heard(turn.user.start_ns, turn.user.end_ns)
        if heard is not None and (url := heard.wav_data_uri(max_length)) is not None:
            span.set_attribute(_INPUT_AUDIO_URL, url)
            span.set_attribute(_INPUT_AUDIO_MIME_TYPE, _WAV)
    spoken = audio.spoken(turn.start_ns, turn.end_ns)
    if spoken is not None and not hide_output_audio(config):
        if (url := spoken.wav_data_uri(max_length)) is not None:
            span.set_attribute(_OUTPUT_AUDIO_URL, url)
            span.set_attribute(_OUTPUT_AUDIO_MIME_TYPE, _WAV)
        if turn.spoken_text is not None:
            span.set_attribute(_OUTPUT_AUDIO_TRANSCRIPT, turn.spoken_text)


def _render_user(
    user: Utterance, tracer: OITracer, parent: context_api.Context, config: TraceConfig
) -> None:
    span = tracer.start_span(
        "user", context=parent, attributes={_SPAN_KIND: "USER"}, start_time=user.start_ns
    )
    try:
        if not hide_input_audio(config):
            span.set_attribute(_INPUT_AUDIO_TRANSCRIPT, user.transcript)
        if user.stt_model is not None:
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, user.stt_model)
        if user.stt_provider is not None:
            span.set_attribute(SpanAttributes.LLM_PROVIDER, user.stt_provider)
    finally:
        _end(span, user.end_ns, failed=False)


def _render_step(
    turn: VoiceTurn,
    step: ModelStep,
    tracer: OITracer,
    parent: context_api.Context,
    config: TraceConfig,
) -> None:
    span = tracer.start_span(
        "assistant", context=parent, attributes={_SPAN_KIND: "LLM"}, start_time=step.start_ns
    )
    try:
        _set_step_attributes(span, turn, step, config)
        step_context = set_span_in_context(span, context_api.Context())
        for tool in turn.tools:
            if turn.step_for(tool) is step:
                _render_tool(tool, tracer, step_context)
    finally:
        _end(span, step.end_ns, failed=False)


def _set_step_attributes(span: Span, turn: VoiceTurn, step: ModelStep, config: TraceConfig) -> None:
    attributes: dict[str, Any] = {}
    if step.model is not None:
        attributes[SpanAttributes.LLM_MODEL_NAME] = step.model
    if step.provider is not None:
        attributes[SpanAttributes.LLM_PROVIDER] = step.provider
    attributes[SpanAttributes.INPUT_VALUE] = messages_json(list(step.input_messages))
    attributes[SpanAttributes.INPUT_MIME_TYPE] = _JSON
    attributes.update(get_llm_input_message_attributes(list(step.input_messages)))
    output = Message(role="assistant")
    if step.output_text:
        output["content"] = step.output_text
        attributes[SpanAttributes.OUTPUT_VALUE] = step.output_text
        attributes[SpanAttributes.OUTPUT_MIME_TYPE] = _TEXT
        if step is _spoken_step(turn) and turn.spoken_text and not hide_output_audio(config):
            attributes[_OUTPUT_AUDIO_TRANSCRIPT] = turn.spoken_text
    if step.tool_calls:
        output["tool_calls"] = [
            ToolCall(id=call.call_id, function={"name": call.name, "arguments": call.arguments})
            for call in step.tool_calls
        ]
        if not step.output_text:
            attributes[SpanAttributes.OUTPUT_VALUE] = messages_json([output])
            attributes[SpanAttributes.OUTPUT_MIME_TYPE] = _JSON
    attributes.update(get_llm_output_message_attributes([output]))
    if step.input_tokens is not None:
        attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = step.input_tokens
    if step.output_tokens is not None:
        attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = step.output_tokens
    if step.time_to_first_token_s is not None:
        attributes[_TIME_TO_FIRST_TOKEN_MS] = int(step.time_to_first_token_s * 1000)
    span.set_attributes(attributes)


def _spoken_step(turn: VoiceTurn) -> ModelStep | None:
    return next((step for step in reversed(turn.steps) if step.output_text), None)


def _render_tool(tool: ToolRun, tracer: OITracer, parent: context_api.Context) -> None:
    span = tracer.start_span(
        tool.name or "tool",
        context=parent,
        attributes={_SPAN_KIND: "TOOL", SpanAttributes.TOOL_NAME: tool.name},
        start_time=tool.start_ns,
    )
    try:
        span.set_attribute(SpanAttributes.INPUT_VALUE, tool.arguments)
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, _JSON)
        if tool.output is not None:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, tool.output)
            span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, _TEXT)
    finally:
        _end(span, tool.end_ns, failed=tool.is_error)


def _end(span: Span, end_ns: int, *, failed: bool) -> None:
    try:
        span.set_status(Status(StatusCode.ERROR if failed else StatusCode.OK))
        span.end(end_time=end_ns)
    except Exception:
        logger.exception("Failed to end LiveKit span")


def _base64_audio_max_length() -> int:
    raw = os.environ.get(_ENV_BASE64_AUDIO_MAX_LENGTH)
    try:
        return int(raw) if raw is not None else _DEFAULT_BASE64_AUDIO_MAX_LENGTH
    except ValueError:
        return _DEFAULT_BASE64_AUDIO_MAX_LENGTH


def hide_input_audio(config: TraceConfig) -> bool:
    return bool(config.hide_inputs) or _env_bool("OPENINFERENCE_HIDE_INPUT_AUDIO")


def hide_output_audio(config: TraceConfig) -> bool:
    return bool(config.hide_outputs) or _env_bool("OPENINFERENCE_HIDE_OUTPUT_AUDIO")


def _env_bool(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}
