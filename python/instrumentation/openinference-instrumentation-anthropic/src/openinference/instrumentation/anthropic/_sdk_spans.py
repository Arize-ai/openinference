"""
anthropic>=1.14.0 starts a span of its own for each API call, on the global tracer provider unless
the client is given another. A call that the instrumentor traces would then have two spans for one
model call, and the SDK's span of a messages.stream() call would not even be in the LLM span's
trace, since the request is sent after stream() has returned.

While the LLM span of a traced call is current, the SDK is handed a non-recording span with the
LLM span's context in place of the span it would start. That is what the SDK gets when a sampler
drops its span: nothing is recorded, the request carries the LLM span's trace context if the
client propagates one, and the spans of HTTP instrumentation are children of the LLM span. Any
other call keeps the SDK's span, and so does a call whose LLM span is not recording.

A tool runner has the same two spans: one for the run and one for each tool that it calls. While
the span of a traced runner is current, the SDK's are replaced in the same way.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, ContextManager, Iterator, Mapping, Optional, Tuple

from opentelemetry import trace as trace_api

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# the LLM span of the traced call whose request is being sent
_llm_span: ContextVar[Optional[trace_api.Span]] = ContextVar("llm_span", default=None)
# the span of the traced tool runner whose step is running
_tool_runner_span: ContextVar[Optional[trace_api.Span]] = ContextVar(
    "tool_runner_span", default=None
)


@contextmanager
def _sending_request_of(span: trace_api.Span) -> Iterator[None]:
    """Makes `span` current for the block, in which the request of the call it traces is sent."""
    token = _llm_span.set(span)
    try:
        with trace_api.use_span(
            span, end_on_exit=False, record_exception=False, set_status_on_exception=False
        ):
            yield
    finally:
        _llm_span.reset(token)


class _StartApiCallSpanWrapper:
    """Wraps the function with which the SDK's client starts the span of an API call."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        try:
            span = _llm_span.get()
            # not for an API call that something else made in the block, e.g. a middleware that
            # started a span of its own
            if span is None or span is not trace_api.get_current_span():
                return wrapped(*args, **kwargs)
            is_recording = span.is_recording()
            span_context = span.get_span_context()
        except Exception:
            logger.exception("Failed to check the span of the API call")
            return wrapped(*args, **kwargs)
        if not is_recording:
            # no LLM span is recorded, e.g. the instrumentor's tracer provider makes none, so the
            # SDK's span is not a second one. Validity does not tell: a no-op tracer hands out the
            # caller's span context
            return wrapped(*args, **kwargs)
        return trace_api.NonRecordingSpan(span_context)


@contextmanager
def _running_step_of(span: trace_api.Span) -> Iterator[None]:
    """Makes the `span` of a tool runner current for the block, in which the runner runs a step."""
    token = _tool_runner_span.set(span)
    try:
        with trace_api.use_span(
            span, end_on_exit=False, record_exception=False, set_status_on_exception=False
        ):
            yield
    finally:
        _tool_runner_span.reset(token)


def _traced_runner_span() -> Optional[trace_api.SpanContext]:
    """The context of the span of the traced tool runner that is running a step, else None."""
    try:
        span = _tool_runner_span.get()
        if span is None or not span.is_recording():
            return None
        span_context = span.get_span_context()
        # by context: the SDK's runner makes a non-recording span with it current in its steps
        if trace_api.get_current_span().get_span_context() != span_context:
            return None
        return span_context
    except Exception:
        logger.exception("Failed to check the span of the tool runner")
        return None


class _StartToolRunnerSpanWrapper:
    """Wraps the function with which the SDK's tool runner starts the span of a run."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if (span_context := _traced_runner_span()) is None:
            return wrapped(*args, **kwargs)
        return trace_api.NonRecordingSpan(span_context)


class _ToolUseSpanWrapper:
    """Wraps the function with which the SDK's tool runner starts the span of a tool call."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> ContextManager[Any]:
        if _traced_runner_span() is None:
            return wrapped(*args, **kwargs)  # type: ignore[no-any-return]
        return _untraced_tool_use()


@contextmanager
def _untraced_tool_use() -> Iterator[Any]:
    # what the SDK's function yields for a client that has no tracer: the run to fill in
    from anthropic.lib._tracing import ToolRun

    yield ToolRun()
