import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Tuple

from opentelemetry import trace as trace_api
from opentelemetry.trace import Span

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _WithSpan:
    """Context manager for managing spans."""

    def __init__(self, span: Optional[Span]) -> None:
        self.span = span

    def __enter__(self) -> "_WithSpan":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.span is not None:
            if exc_type is not None:
                self.span.set_status(trace_api.Status(trace_api.StatusCode.ERROR))
                self.span.record_exception(exc_val)
            else:
                self.span.set_status(trace_api.Status(trace_api.StatusCode.OK))
            self.span.end()


@contextmanager
def _with_span(
    tracer: trace_api.Tracer,
    name: str,
    attributes: Tuple[Tuple[str, Any], ...],
) -> Iterator[Optional[Span]]:
    """Create a span with the given name and attributes."""
    span = tracer.start_span(name)
    if span is None:
        yield None
        return

    try:
        for key, value in attributes:
            if value is not None:
                span.set_attribute(key, value)
        yield span
    finally:
        span.end() 