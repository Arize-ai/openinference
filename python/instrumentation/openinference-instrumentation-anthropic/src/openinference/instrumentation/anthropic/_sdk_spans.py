"""
anthropic>=1.14.0 starts a span of its own for each API call, on the global tracer provider unless
the client is given another. A call that the instrumentor traces would then have two spans for one
model call, and the SDK's span of a messages.stream() call would not even be in the LLM span's
trace, since the request is sent after stream() has returned.

While the LLM span of a traced call is current, the SDK is handed a non-recording span with the
LLM span's context in place of the span it would start. That is what the SDK gets when a sampler
drops its span: nothing is recorded, the request carries the LLM span's trace context if the
client propagates one, and the spans of HTTP instrumentation are children of the LLM span. Any
other call keeps the SDK's span.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator, Mapping, Optional, Tuple

from opentelemetry import trace as trace_api

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# the LLM span of the traced call whose request is being sent
_llm_span: ContextVar[Optional[trace_api.Span]] = ContextVar("llm_span", default=None)


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
            span_context = span.get_span_context()
        except Exception:
            logger.exception("Failed to check the span of the API call")
            return wrapped(*args, **kwargs)
        if not span_context.is_valid:
            # what the SDK returns for a span with no effect: the call then has no span
            return None
        return trace_api.NonRecordingSpan(span_context)
