from __future__ import annotations

import logging
import os
from typing import Any

from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY, get_value, set_value
from opentelemetry.trace import (
    Link,
    Span,
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
    set_span_in_context,
)

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    get_attributes_from_context,
    get_llm_input_message_attributes,
    get_llm_output_message_attributes,
)
from openinference.instrumentation.livekit_agents._fold import ModelStep, ToolCall, VoiceTurn
from openinference.instrumentation.livekit_agents._messages import oi_messages
from openinference.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_AUDIO_KIND = "AUDIO"
_USER_KIND = "USER"
_LLM_KIND = "LLM"
_TOOL_KIND = "TOOL"
_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
_END_REASON = "end_reason"
_TIME_TO_FIRST_TOKEN_MS = "time_to_first_token_ms"
_INPUT_AUDIO_TRANSCRIPT = "input.audio.transcript"
_OUTPUT_AUDIO_TRANSCRIPT = "output.audio.transcript"

_TURN_NAME = "conversation.turn"
_USER_NAME = "user"
_LLM_NAME = "assistant"


def render_turn(turn: VoiceTurn, tracer: OITracer, config: TraceConfig) -> None:
    if get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return

    turn_span: Span | None = None
    token = None
    try:
        context = context_api.Context()
        context_attributes = dict(get_attributes_from_context())
        if SpanAttributes.SESSION_ID not in context_attributes and turn.session_id is not None:
            context_attributes[SpanAttributes.SESSION_ID] = turn.session_id
        for key, value in context_attributes.items():
            context = set_value(key, value, context)
        token = context_api.attach(context)

        turn_span = tracer.start_span(
            _TURN_NAME,
            context=context_api.Context(),
            attributes={_SPAN_KIND: _AUDIO_KIND},
            links=[_audio_link(turn)],
            start_time=turn.start_ns,
        )
        try:
            _set_turn_attributes(turn_span, turn)
        except Exception:
            logger.exception("Failed to set LiveKit conversation turn attributes")

        turn_context = set_span_in_context(turn_span, context_api.Context())
        _render_user(turn, tracer, turn_context, config)
        for step in turn.steps:
            _render_step(turn, step, tracer, turn_context, config)
        for tool in turn.tools:
            if tool.step_span_id is None:
                _render_tool(tool, tracer, turn_context)
    except Exception:
        logger.exception("Failed to render LiveKit conversation turn")
    finally:
        if turn_span is not None:
            try:
                turn_span.set_status(_status(turn.failed))
                turn_span.end(end_time=turn.end_ns)
            except Exception:
                logger.exception("Failed to end LiveKit conversation turn")
        if token is not None:
            context_api.detach(token)


def _set_turn_attributes(span: Span, turn: VoiceTurn) -> None:
    span.set_attribute(_END_REASON, turn.end_reason)
    model = next((step.model for step in turn.steps if step.model), None)
    if model is not None:
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model)
    input_value = turn.user.transcript if turn.user is not None else turn.typed_text
    if input_value:
        span.set_attribute(SpanAttributes.INPUT_VALUE, input_value)
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "text/plain")
    output_value = "\n".join(
        step.spoken_text for step in turn.steps if step.spoken_text is not None
    )
    if output_value:
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, output_value)
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")


def _render_user(
    turn: VoiceTurn,
    tracer: OITracer,
    parent_context: context_api.Context,
    config: TraceConfig,
) -> None:
    if turn.user is None and not turn.typed_text:
        return
    start_ns = turn.user.start_ns if turn.user is not None else turn.start_ns
    end_ns = turn.user.end_ns if turn.user is not None else turn.end_ns
    span: Span | None = None
    try:
        span = tracer.start_span(
            _USER_NAME,
            context=parent_context,
            attributes={_SPAN_KIND: _USER_KIND},
            start_time=start_ns,
        )
        if turn.user is not None and not _hide_input_audio(config):
            span.set_attribute(_INPUT_AUDIO_TRANSCRIPT, turn.user.transcript)
        elif turn.typed_text and not config.hide_inputs:
            span.set_attribute(SpanAttributes.INPUT_VALUE, turn.typed_text)
            span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "text/plain")
        if turn.user is not None and turn.user.stt_model is not None:
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, turn.user.stt_model)
        if turn.user is not None and turn.user.stt_provider is not None:
            span.set_attribute(SpanAttributes.LLM_PROVIDER, turn.user.stt_provider)
        span.set_status(_status(False))
    except Exception:
        logger.exception("Failed to render LiveKit user span")
    finally:
        if span is not None:
            _end(span, end_ns, _USER_NAME)


def _render_step(
    turn: VoiceTurn,
    step: ModelStep,
    tracer: OITracer,
    parent_context: context_api.Context,
    config: TraceConfig,
) -> None:
    span: Span | None = None
    try:
        span = tracer.start_span(
            _LLM_NAME,
            context=parent_context,
            attributes={_SPAN_KIND: _LLM_KIND},
            start_time=step.start_ns,
        )
        _set_step_attributes(span, step, config)
        step_context = set_span_in_context(span, context_api.Context())
        for tool in turn.tools:
            if tool.step_span_id == step.span_id:
                _render_tool(tool, tracer, step_context)
        span.set_status(_status(False))
    except Exception:
        logger.exception("Failed to render LiveKit assistant span")
    finally:
        if span is not None:
            _end(span, step.end_ns, _LLM_NAME)


def _set_step_attributes(span: Span, step: ModelStep, config: TraceConfig) -> None:
    attributes: dict[str, Any] = {}
    if step.model is not None:
        attributes[SpanAttributes.LLM_MODEL_NAME] = step.model
    if step.provider is not None:
        attributes[SpanAttributes.LLM_PROVIDER] = step.provider
    if step.input_messages_json is not None:
        attributes[SpanAttributes.INPUT_VALUE] = step.input_messages_json
        attributes[SpanAttributes.INPUT_MIME_TYPE] = "application/json"
    if step.spoken_text is not None:
        attributes[SpanAttributes.OUTPUT_VALUE] = step.spoken_text
        attributes[SpanAttributes.OUTPUT_MIME_TYPE] = "text/plain"
        if not _hide_output_audio(config):
            attributes[_OUTPUT_AUDIO_TRANSCRIPT] = step.spoken_text
    elif step.output_messages_json is not None:
        attributes[SpanAttributes.OUTPUT_VALUE] = step.output_messages_json
        attributes[SpanAttributes.OUTPUT_MIME_TYPE] = "application/json"
    attributes.update(
        get_llm_input_message_attributes(
            oi_messages(step.input_messages_json, step.system_instructions_json)
        )
    )
    attributes.update(get_llm_output_message_attributes(oi_messages(step.output_messages_json)))
    optional_counts = {
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: step.input_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: step.output_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO: step.audio_input_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO: step.audio_output_tokens,
    }
    attributes.update({key: value for key, value in optional_counts.items() if value is not None})
    if step.time_to_first_token_s is not None:
        attributes[_TIME_TO_FIRST_TOKEN_MS] = int(step.time_to_first_token_s * 1000)
    for key, value in attributes.items():
        span.set_attribute(key, value)


def _render_tool(
    tool: ToolCall,
    tracer: OITracer,
    parent_context: context_api.Context,
) -> None:
    name = _tool_name(tool)
    span: Span | None = None
    try:
        span = tracer.start_span(
            name,
            context=parent_context,
            attributes={_SPAN_KIND: _TOOL_KIND},
            start_time=tool.start_ns,
        )
        span.set_attribute(SpanAttributes.TOOL_NAME, name)
        if tool.arguments is not None:
            span.set_attribute(SpanAttributes.INPUT_VALUE, tool.arguments)
            span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
        if tool.output is not None:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, tool.output)
            span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")
        span.set_status(_status(tool.is_error))
    except Exception:
        logger.exception("Failed to render LiveKit tool span")
    finally:
        if span is not None:
            _end(span, tool.end_ns, name)


def _audio_link(turn: VoiceTurn) -> Link:
    return Link(
        SpanContext(
            trace_id=turn.trace_id,
            span_id=turn.agent_turn_span_id,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
    )


def _status(failed: bool) -> Status:
    if failed:
        return Status(StatusCode.ERROR)
    return Status(StatusCode.OK)


def _hide_input_audio(config: TraceConfig) -> bool:
    return bool(config.hide_inputs) or _env_bool("OPENINFERENCE_HIDE_INPUT_AUDIO")


def _hide_output_audio(config: TraceConfig) -> bool:
    return bool(config.hide_outputs) or _env_bool("OPENINFERENCE_HIDE_OUTPUT_AUDIO")


def _env_bool(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _end(span: Span, end_ns: int, name: str) -> None:
    try:
        span.end(end_time=end_ns)
    except Exception:
        logger.exception("Failed to end LiveKit rendered span %s", name)


def _tool_name(tool: ToolCall) -> str:
    return tool.name or "tool"
