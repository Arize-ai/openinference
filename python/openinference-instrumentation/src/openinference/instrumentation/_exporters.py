import os
import threading
from typing import Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import SpanContext, SpanKind, TraceFlags


class EnsureRootSpanExporter(SpanExporter):
    """
    A SpanExporter wrapper that ensures every exported trace contains at least
    one root span (a span with no parent).

    Braintrust's logs table only displays traces that contain a root span --
    a span whose ``parent_span_id`` is empty. If all spans in a trace have a
    parent (e.g. because a ``TRACEPARENT`` environment variable is set or the
    instrumented code runs inside an existing distributed trace), the entire
    trace is invisible in Braintrust's UI.

    Wrap your Braintrust OTLP exporter with this class to automatically inject
    a minimal synthetic root span whenever one is missing:

    .. code-block:: python

        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from openinference.instrumentation import EnsureRootSpanExporter

        exporter = EnsureRootSpanExporter(OTLPSpanExporter(...))
        # Use `exporter` as the exporter argument to BatchSpanProcessor, etc.

    A synthetic root span is created at most once per ``trace_id``.  If a root
    span for a given trace is exported in a later batch the synthetic one has
    already been emitted, which is acceptable: Braintrust will simply display
    two root-level entries for that trace.
    """

    def __init__(self, wrapped: SpanExporter) -> None:
        self._wrapped = wrapped
        self._root_trace_ids: set[int] = set()
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        synthetic_roots: list[ReadableSpan] = []

        with self._lock:
            # First pass: record any real root spans present in this batch so
            # we do not create a duplicate for them.
            for span in spans:
                if span.parent is None and span.context is not None:
                    self._root_trace_ids.add(span.context.trace_id)

            # Second pass: for each trace_id that still has no root span,
            # synthesize one from the first child span we encounter.
            for span in spans:
                if span.parent is not None and span.context is not None:
                    trace_id = span.context.trace_id
                    if trace_id not in self._root_trace_ids:
                        self._root_trace_ids.add(trace_id)
                        synthetic_roots.append(_make_root_span(span))

        return self._wrapped.export(synthetic_roots + list(spans))

    def shutdown(self) -> None:
        self._wrapped.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._wrapped.force_flush(timeout_millis)


def _make_root_span(child: ReadableSpan) -> ReadableSpan:
    """Return a minimal root span that shares *child*'s trace_id."""
    assert child.context is not None  # guarded by caller

    span_id = int.from_bytes(os.urandom(8), "big")
    context = SpanContext(
        trace_id=child.context.trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(child.context.trace_flags),
        trace_state=child.context.trace_state,
    )
    return ReadableSpan(
        name=child.name,
        context=context,
        parent=None,
        resource=child.resource,
        kind=SpanKind.INTERNAL,
        instrumentation_scope=child.instrumentation_scope,
        start_time=child.start_time,
        end_time=child.end_time,
    )


# Re-export for convenience so callers can type-annotate against the Optional
# return type of _make_root_span without importing from the private module.
__all__ = ["EnsureRootSpanExporter"]
