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


class _RealtimeSessionWrapper(ObjectProxy):  # type: ignore[misc]
    """Wraps RealtimeSession to intercept events and create OTel spans.

    This wrapper intercepts the async iterator protocol of RealtimeSession
    to observe events and create corresponding OpenInference spans.
    """

    __slots__ = (
        "_self_tracer",
        "_self_suppressed",
        "_self_iterator",
        "_self_root_span",
        "_self_root_token",
        "_self_agent_span",
        "_self_agent_token",
        "_self_agent_error_msg",
        "_self_tool_spans",
        "_self_tool_tokens",
    )

    def __init__(self, wrapped: Any, tracer: Tracer) -> None:
        super().__init__(wrapped)
        self._self_tracer = tracer
        self._self_suppressed = False
        # RealtimeSession.__aiter__ is an async generator function — calling it returns
        # an async generator object that has __anext__. We store it on first __aiter__ call.
        self._self_iterator: Optional[Any] = None
        self._self_root_span: Optional[OtelSpan] = None
        self._self_root_token: Optional[object] = None
        self._self_agent_span: Optional[OtelSpan] = None
        self._self_agent_token: Optional[object] = None
        self._self_agent_error_msg: Optional[str] = None
        self._self_tool_spans: dict[str, list[OtelSpan]] = {}
        self._self_tool_tokens: dict[str, list[object]] = {}

    async def __aenter__(self) -> _RealtimeSessionWrapper:
        await self.__wrapped__.__aenter__()

        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            self._self_suppressed = True
            return self

        try:
            agent_name = getattr(self.__wrapped__, "_agent", None)
            name = getattr(agent_name, "name", None) if agent_name else None
            span_name = f"RealtimeSession: {name}" if name else "RealtimeSession"

            attributes: dict[str, Any] = {
                OPENINFERENCE_SPAN_KIND: AGENT,
                LLM_SYSTEM: "openai",
            }
            if name:
                attributes[GRAPH_NODE_ID] = name
            for k, v in get_attributes_from_context():
                attributes[k] = v

            self._self_root_span = self._self_tracer.start_span(
                name=span_name,
                attributes=attributes,
            )
            self._self_root_token = attach(set_span_in_context(self._self_root_span))
        except Exception:
            logger.exception("Error starting realtime session span")

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
                    target_span = self._self_agent_span or self._self_root_span
                    if target_span is not None and target_span.is_recording():
                        target_span.record_exception(exc)
                        target_span.set_status(
                            Status(StatusCode.ERROR, description=exc_description)
                        )
                        if target_span is self._self_agent_span:
                            self._self_agent_error_msg = exc_description
                except Exception:
                    logger.exception("Error recording exception on realtime span")
            raise

        if self._self_suppressed or self._self_root_span is None:
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
        # audio, history, raw_model_event, input_audio_timeout_triggered — no spans (too noisy)

    def _on_agent_start(self, event: Any) -> None:
        agent = getattr(event, "agent", None)
        name = getattr(agent, "name", None) or "agent"

        # End any previous agent span first (consecutive agents without explicit end)
        self._end_agent_span(error=self._self_agent_error_msg)

        parent_span = self._self_root_span
        context = set_span_in_context(parent_span) if parent_span else None
        span = self._self_tracer.start_span(
            name=f"Agent: {name}",
            context=context,
            attributes={
                OPENINFERENCE_SPAN_KIND: AGENT,
                LLM_SYSTEM: "openai",
                GRAPH_NODE_ID: name,
            },
        )
        self._self_agent_span = span
        self._self_agent_token = attach(set_span_in_context(span))

    def _on_agent_end(self, event: Any) -> None:
        self._end_agent_span(error=self._self_agent_error_msg)

    def _end_agent_span(self, error: Optional[str] = None) -> None:
        if self._self_agent_span is None:
            return
        # End any open tool spans first
        self._end_all_tool_spans(ok=error is None)
        if self._self_agent_token is not None:
            detach(self._self_agent_token)  # type: ignore[arg-type]
            self._self_agent_token = None
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

        # Use agent span as parent if available, else root
        parent_span = self._self_agent_span or self._self_root_span
        context = set_span_in_context(parent_span) if parent_span else None
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

        parent_span = self._self_agent_span or self._self_root_span
        context = set_span_in_context(parent_span) if parent_span else None
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

        # Set error status on whichever span is currently active
        if self._self_agent_span is not None:
            self._self_agent_span.set_status(Status(StatusCode.ERROR, description=error_msg))
            self._self_agent_error_msg = error_msg
        elif self._self_root_span is not None:
            self._self_root_span.set_status(Status(StatusCode.ERROR, description=error_msg))

    def _on_guardrail_tripped(self, event: Any) -> None:
        results = getattr(event, "guardrail_results", None)
        name = "guardrail"
        if results:
            first = results[0] if results else None
            guardrail = getattr(first, "guardrail", None) if first else None
            guardrail_name = getattr(guardrail, "name", None) if guardrail else None
            if guardrail_name:
                name = guardrail_name

        parent_span = self._self_agent_span or self._self_root_span
        context = set_span_in_context(parent_span) if parent_span else None
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
        # If session exited with exception, propagate it; otherwise use any stored error
        exc_error = str(exc_val) if exc_val else None
        # _end_agent_span also ends tool spans; call directly in case there's no agent span
        self._end_all_tool_spans(ok=exc_error is None)
        self._end_agent_span(error=exc_error or self._self_agent_error_msg)

        if self._self_root_span is not None:
            if self._self_root_token is not None:
                detach(self._self_root_token)  # type: ignore[arg-type]
                self._self_root_token = None
            if exc_val is not None:
                self._self_root_span.set_status(Status(StatusCode.ERROR, description=str(exc_val)))
            else:
                self._self_root_span.set_status(Status(StatusCode.OK))
            self._self_root_span.end()
            self._self_root_span = None


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
