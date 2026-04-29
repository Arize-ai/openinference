from __future__ import annotations

import logging
from typing import Any, Optional

import opentelemetry.context as context_api
from opentelemetry.context import attach, detach
from opentelemetry.trace import Span as OtelSpan
from opentelemetry.trace import Status, StatusCode, Tracer, set_span_in_context
from wrapt import ObjectProxy, wrap_function_wrapper

from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

logger = logging.getLogger(__name__)

OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
GRAPH_NODE_ID = SpanAttributes.GRAPH_NODE_ID
GRAPH_NODE_PARENT_ID = SpanAttributes.GRAPH_NODE_PARENT_ID
TOOL_NAME = SpanAttributes.TOOL_NAME
INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE

AGENT = OpenInferenceSpanKindValues.AGENT.value
TOOL = OpenInferenceSpanKindValues.TOOL.value
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
JSON = OpenInferenceMimeTypeValues.JSON.value
TEXT = OpenInferenceMimeTypeValues.TEXT.value

_SUPPRESS_INSTRUMENTATION_KEY = context_api._SUPPRESS_INSTRUMENTATION_KEY


def _extract_item_text(item: Any) -> str:
    """Extract text/transcript from a RealtimeMessageItem's content list."""
    content_list = getattr(item, "content", None) or []
    parts = []
    for content in content_list:
        content_type = getattr(content, "type", None)
        if content_type in ("input_text", "text"):
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
        elif content_type in ("input_audio", "audio"):
            transcript = getattr(content, "transcript", None)
            if transcript:
                parts.append(transcript)
    return " ".join(parts)


class _RealtimeSessionWrapper(ObjectProxy):  # type: ignore[misc]
    """Wraps RealtimeSession to intercept events and create OTel spans.

    This wrapper intercepts the async iterator protocol of RealtimeSession
    to observe events and create corresponding OpenInference spans.

    Span structure:
        Agent: <name>          [AGENT - one per session, reused across turns]
          Tool: <tool_name>    [TOOL  - one per tool call]
          Handoff: A -> B      [TOOL  - one per handoff]
          Guardrail: <name>    [CHAIN - one per triggered guardrail]
    """

    __slots__ = (
        "_self_tracer",
        "_self_suppressed",
        "_self_iterator",
        "_self_session_context",
        "_self_agent_name",
        "_self_agent_span",
        "_self_agent_token",
        "_self_agent_error_msg",
        "_self_tool_spans",
        "_self_tool_tokens",
        "_self_last_user_text",
        "_self_agent_output",
        "_self_transcript_parts",
    )

    def __init__(self, wrapped: Any, tracer: Tracer) -> None:
        super().__init__(wrapped)
        self._self_tracer = tracer
        self._self_suppressed = False
        # RealtimeSession.__aiter__ is an async generator function — calling it returns
        # an async generator object that has __anext__. We store it on first __aiter__ call.
        self._self_iterator: Optional[Any] = None
        self._self_session_context: Optional[object] = None  # trace context from first agent span
        self._self_agent_name: Optional[str] = None
        self._self_agent_span: Optional[OtelSpan] = None
        self._self_agent_token: Optional[object] = None
        self._self_agent_error_msg: Optional[str] = None
        self._self_tool_spans: dict[str, list[OtelSpan]] = {}
        self._self_tool_tokens: dict[str, list[object]] = {}
        self._self_last_user_text: Optional[str] = None
        self._self_agent_output: Optional[str] = None
        self._self_transcript_parts: list[str] = []

    async def __aenter__(self) -> _RealtimeSessionWrapper:
        await self.__wrapped__.__aenter__()

        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            self._self_suppressed = True
            return self

        try:
            agent_obj = getattr(self.__wrapped__, "_current_agent", None)
            name = getattr(agent_obj, "name", None) or "agent"
            self._self_agent_name = name

            attributes: dict[str, Any] = {
                OPENINFERENCE_SPAN_KIND: AGENT,
                LLM_SYSTEM: "openai",
                GRAPH_NODE_ID: name,
            }
            for k, v in get_attributes_from_context():
                attributes[k] = v

            self._self_agent_span = self._self_tracer.start_span(
                name=f"Agent: {name}",
                attributes=attributes,
            )
            ctx = set_span_in_context(self._self_agent_span)
            # Store the session-level context so handoff spans stay in the same trace
            self._self_session_context = ctx
            self._self_agent_token = attach(ctx)
        except Exception:
            logger.exception("Error starting realtime agent span")

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if not self._self_suppressed:
                self._end_all_open_spans(exc_val)
        except Exception:
            logger.exception("Error ending realtime session spans")
        finally:
            await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb)

    def __aiter__(self) -> _RealtimeSessionWrapper:
        # Materialise the async generator from the wrapped session on first call.
        # RealtimeSession.__aiter__ is an async generator function, so calling it
        # returns an async generator object with __anext__. We must capture it here
        # rather than trying to call __anext__ directly on the session.
        if self._self_iterator is None:
            self._self_iterator = self.__wrapped__.__aiter__()
        return self

    async def __anext__(self) -> Any:
        if self._self_iterator is None:
            # Should not happen in normal usage, but be safe
            self._self_iterator = self.__wrapped__.__aiter__()
        try:
            event = await self._self_iterator.__anext__()
        except StopAsyncIteration:
            raise
        except Exception as exc:
            # Unexpected error from the underlying session — record it and propagate
            if not self._self_suppressed:
                try:
                    exc_description = f"{type(exc).__name__}: {exc}"
                    if self._self_agent_span is not None and self._self_agent_span.is_recording():
                        self._self_agent_span.record_exception(exc)
                        self._self_agent_span.set_status(
                            Status(StatusCode.ERROR, description=exc_description)
                        )
                        self._self_agent_error_msg = exc_description
                except Exception:
                    logger.exception("Error recording exception on realtime span")
            raise

        if self._self_suppressed:
            return event

        try:
            self._dispatch_event(event)
        except Exception:
            logger.exception("Error processing realtime event for tracing")

        return event

    def _dispatch_event(self, event: Any) -> None:
        event_type = getattr(event, "type", None)

        if event_type == "agent_start":
            self._on_agent_start(event)
        elif event_type == "agent_end":
            self._on_agent_end(event)
        elif event_type == "tool_start":
            self._on_tool_start(event)
        elif event_type == "tool_end":
            self._on_tool_end(event)
        elif event_type == "handoff":
            self._on_handoff(event)
        elif event_type == "error":
            self._on_error(event)
        elif event_type == "guardrail_tripped":
            self._on_guardrail_tripped(event)
        elif event_type == "history_added":
            self._on_history_added(event)
        elif event_type == "history_updated":
            self._on_history_updated(event)
        elif event_type == "raw_model_event":
            self._on_raw_model_event(event)
        # audio, input_audio_timeout_triggered — no spans (too noisy)

    def _on_agent_start(self, event: Any) -> None:
        agent = getattr(event, "agent", None)
        name = getattr(agent, "name", None) or "agent"

        # If the same agent is already active, skip — the SDK fires agent_start on every
        # response.created (including after tool output), which would create duplicate spans.
        if self._self_agent_span is not None and name == self._self_agent_name:
            return

        # Different agent (handoff) — end the current span and start a new one.
        self._end_agent_span(error=self._self_agent_error_msg)

        self._self_agent_name = name
        self._self_agent_output = None
        self._self_transcript_parts = []

        # Parent to the session context so all agents share the same trace
        span = self._self_tracer.start_span(
            name=f"Agent: {name}",
            context=self._self_session_context,  # type: ignore[arg-type]
            attributes={
                OPENINFERENCE_SPAN_KIND: AGENT,
                LLM_SYSTEM: "openai",
                GRAPH_NODE_ID: name,
            },
        )
        self._self_agent_span = span
        self._self_agent_token = attach(set_span_in_context(span))

    def _on_agent_end(self, event: Any) -> None:
        # agent_end fires on every turn_ended — don't close the span here since
        # the same agent may continue in the next turn. The span is closed in
        # __aexit__ or when a handoff triggers a different agent.
        pass

    def _end_agent_span(self, error: Optional[str] = None) -> None:
        if self._self_agent_span is None:
            return
        # End any open tool spans first
        self._end_all_tool_spans(ok=error is None)
        if self._self_agent_token is not None:
            detach(self._self_agent_token)  # type: ignore[arg-type]
            self._self_agent_token = None
        if self._self_last_user_text is not None:
            self._self_agent_span.set_attribute(INPUT_VALUE, self._self_last_user_text)
            self._self_agent_span.set_attribute(INPUT_MIME_TYPE, TEXT)
        # Prefer transcript accumulated from transcript_delta events (available before agent_end)
        # Fall back to history-based output (for text responses)
        output = (
            "".join(self._self_transcript_parts)
            if self._self_transcript_parts
            else self._self_agent_output
        )
        if output:
            self._self_agent_span.set_attribute(OUTPUT_VALUE, output)
            self._self_agent_span.set_attribute(OUTPUT_MIME_TYPE, TEXT)
        if error is None:
            self._self_agent_span.set_status(Status(StatusCode.OK))
        else:
            self._self_agent_span.set_status(Status(StatusCode.ERROR, description=error))
        self._self_agent_span.end()
        self._self_agent_span = None
        self._self_agent_error_msg = None

    def _on_tool_start(self, event: Any) -> None:
        tool = getattr(event, "tool", None)
        name = getattr(tool, "name", None) or "tool"

        context = set_span_in_context(self._self_agent_span) if self._self_agent_span else None
        span = self._self_tracer.start_span(
            name=f"Tool: {name}",
            context=context,
            attributes={
                OPENINFERENCE_SPAN_KIND: TOOL,
                LLM_SYSTEM: "openai",
                TOOL_NAME: name,
            },
        )
        self._self_tool_spans.setdefault(name, []).append(span)
        self._self_tool_tokens.setdefault(name, []).append(attach(set_span_in_context(span)))

    def _on_tool_end(self, event: Any) -> None:
        tool = getattr(event, "tool", None)
        name = getattr(tool, "name", None) or "tool"
        output = getattr(event, "output", None)

        spans_list = self._self_tool_spans.get(name)
        tokens_list = self._self_tool_tokens.get(name)

        if not spans_list:
            return
        # LIFO pop: assumes the SDK emits tool_end events in reverse-start order for
        # concurrent calls to the same tool. No call_id correlation field is available
        # on RealtimeToolEnd in the current SDK, so LIFO is the best available heuristic.
        span = spans_list.pop()
        token = tokens_list.pop() if tokens_list else None
        if not spans_list:
            self._self_tool_spans.pop(name, None)
            self._self_tool_tokens.pop(name, None)

        if token is not None:
            detach(token)  # type: ignore[arg-type]
        if output is not None:
            if isinstance(output, str):
                span.set_attribute(OUTPUT_VALUE, output)
                span.set_attribute(OUTPUT_MIME_TYPE, TEXT)
            else:
                span.set_attribute(OUTPUT_VALUE, safe_json_dumps(output))
                span.set_attribute(OUTPUT_MIME_TYPE, JSON)
        span.set_status(Status(StatusCode.OK))
        span.end()

    def _end_all_tool_spans(self, ok: bool = True) -> None:
        for name in list(self._self_tool_spans.keys()):
            spans_list = self._self_tool_spans.pop(name, [])
            tokens_list = self._self_tool_tokens.pop(name, [])
            for token in reversed(tokens_list):
                if token is not None:
                    detach(token)  # type: ignore[arg-type]
            for span in reversed(spans_list):
                if span is not None:
                    span.set_status(Status(StatusCode.OK if ok else StatusCode.ERROR))
                    span.end()

    def _on_handoff(self, event: Any) -> None:
        from_agent = getattr(event, "from_agent", None)
        to_agent = getattr(event, "to_agent", None)
        from_name = getattr(from_agent, "name", None) or "unknown"
        to_name = getattr(to_agent, "name", None) or "unknown"

        context = set_span_in_context(self._self_agent_span) if self._self_agent_span else None
        span = self._self_tracer.start_span(
            name=f"Handoff: {from_name} -> {to_name}",
            context=context,
            attributes={
                OPENINFERENCE_SPAN_KIND: TOOL,
                LLM_SYSTEM: "openai",
                GRAPH_NODE_ID: to_name,
                GRAPH_NODE_PARENT_ID: from_name,
            },
        )
        span.set_status(Status(StatusCode.OK))
        span.end()

    def _on_error(self, event: Any) -> None:
        error = getattr(event, "error", None)
        error_msg = str(error) if error is not None else "Realtime error"

        if self._self_agent_span is not None:
            self._self_agent_span.set_status(Status(StatusCode.ERROR, description=error_msg))
            self._self_agent_error_msg = error_msg

    def _on_raw_model_event(self, event: Any) -> None:
        data = getattr(event, "data", None)
        if data is None:
            return
        # Accumulate assistant audio transcript deltas — these arrive before audio_end
        # and are the only reliable source of spoken output when the user breaks early.
        if getattr(data, "type", None) == "transcript_delta":
            delta = getattr(data, "delta", None)
            if delta:
                self._self_transcript_parts.append(delta)

    def _on_history_added(self, event: Any) -> None:
        item = getattr(event, "item", None)
        if item is None:
            return
        role = getattr(item, "role", None)
        if role != "user":
            return
        text = _extract_item_text(item)
        if text:
            self._self_last_user_text = text

    def _on_history_updated(self, event: Any) -> None:
        history = getattr(event, "history", None)
        if not history:
            return
        # Update last user text (audio transcript may arrive asynchronously after history_added)
        for item in reversed(history):
            if getattr(item, "role", None) == "user":
                text = _extract_item_text(item)
                if text:
                    self._self_last_user_text = text
                break
        # Capture assistant text output (for text-modality responses — the SDK sets
        # status="in_progress" even on done events, so we accept any assistant text)
        if not self._self_transcript_parts:
            for item in reversed(history):
                if getattr(item, "role", None) == "assistant":
                    text = _extract_item_text(item)
                    if text:
                        self._self_agent_output = text
                    break

    def _on_guardrail_tripped(self, event: Any) -> None:
        results = getattr(event, "guardrail_results", None)
        name = "guardrail"
        if results:
            first = results[0] if results else None
            guardrail = getattr(first, "guardrail", None) if first else None
            guardrail_name = getattr(guardrail, "name", None) if guardrail else None
            if guardrail_name:
                name = guardrail_name

        context = set_span_in_context(self._self_agent_span) if self._self_agent_span else None
        span = self._self_tracer.start_span(
            name=f"Guardrail: {name}",
            context=context,
            attributes={
                OPENINFERENCE_SPAN_KIND: CHAIN,
                LLM_SYSTEM: "openai",
            },
        )
        message = getattr(event, "message", None)
        if message:
            span.set_attribute(OUTPUT_VALUE, message)
            span.set_attribute(OUTPUT_MIME_TYPE, TEXT)
        span.set_status(Status(StatusCode.OK))
        span.end()

    def _end_all_open_spans(self, exc_val: Optional[BaseException] = None) -> None:
        """End all open spans when the session exits, in order."""
        exc_error = str(exc_val) if exc_val is not None else None
        # _end_agent_span also ends tool spans
        self._end_agent_span(
            error=exc_error if exc_error is not None else self._self_agent_error_msg
        )


class _RealtimeRunnerRunWrapper:
    """Wraps RealtimeRunner.run() to return a _RealtimeSessionWrapper."""

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> _RealtimeSessionWrapper:
        session = await wrapped(*args, **kwargs)
        return _RealtimeSessionWrapper(session, self._tracer)


def _patch_realtime(tracer: Tracer) -> None:
    """Patch RealtimeRunner.run() to wrap the returned session with tracing."""
    wrap_function_wrapper(
        module="agents.realtime.runner",
        name="RealtimeRunner.run",
        wrapper=_RealtimeRunnerRunWrapper(tracer),
    )


def _unpatch_realtime() -> None:
    """Remove the patch from RealtimeRunner.run()."""
    try:
        from agents.realtime.runner import RealtimeRunner

        run = getattr(RealtimeRunner, "run", None)
        if run is not None and hasattr(run, "__wrapped__"):
            RealtimeRunner.run = run.__wrapped__  # type: ignore[method-assign]
    except ImportError:
        pass
