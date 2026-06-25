"""Tests for EnsureRootSpanExporter."""

from typing import List, Optional, Sequence
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext, SpanKind, TraceFlags

from openinference.instrumentation import EnsureRootSpanExporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRACE_ID = 0x000102030405060708090A0B0C0D0E0F
_SPAN_ID_ROOT = 0xAABBCCDDEEFF0011
_SPAN_ID_CHILD = 0x1122334455667788
_SPAN_ID_GRANDCHILD = 0x99AABBCCDDEEFF00


def _make_span_context(trace_id: int, span_id: int) -> SpanContext:
    return SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )


def _make_span(
    name: str,
    trace_id: int,
    span_id: int,
    parent: Optional[SpanContext] = None,
) -> ReadableSpan:
    return ReadableSpan(
        name=name,
        context=_make_span_context(trace_id, span_id),
        parent=parent,
        resource=Resource.create(),
        kind=SpanKind.INTERNAL,
        start_time=1_000_000_000,
        end_time=2_000_000_000,
    )


class _CapturingExporter:
    """Minimal exporter that captures exported spans."""

    def __init__(self) -> None:
        self.exported: List[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.exported.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_root_span_passes_through_unchanged() -> None:
    """A root span (no parent) is forwarded as-is; no synthetic span is added."""
    inner = _CapturingExporter()
    exporter = EnsureRootSpanExporter(inner)

    root = _make_span("root-op", _TRACE_ID, _SPAN_ID_ROOT, parent=None)
    result = exporter.export([root])

    assert result == SpanExportResult.SUCCESS
    assert len(inner.exported) == 1
    assert inner.exported[0] is root


def test_synthetic_root_injected_for_orphan_child() -> None:
    """A child span with no preceding root gets a synthetic root prepended."""
    inner = _CapturingExporter()
    exporter = EnsureRootSpanExporter(inner)

    parent_ctx = _make_span_context(_TRACE_ID, _SPAN_ID_ROOT)
    child = _make_span("child-op", _TRACE_ID, _SPAN_ID_CHILD, parent=parent_ctx)
    exporter.export([child])

    assert len(inner.exported) == 2
    synthetic, original = inner.exported
    # Synthetic root has no parent
    assert synthetic.parent is None
    # Both share the same trace_id
    assert synthetic.context is not None
    assert synthetic.context.trace_id == _TRACE_ID
    # Synthetic span_id is different from the child's
    assert synthetic.context.span_id != _SPAN_ID_CHILD
    # Original child is unchanged
    assert original is child


def test_no_duplicate_synthetic_root_across_batches() -> None:
    """Only one synthetic root is created per trace_id across multiple batches."""
    inner = _CapturingExporter()
    exporter = EnsureRootSpanExporter(inner)

    parent_ctx = _make_span_context(_TRACE_ID, _SPAN_ID_ROOT)
    child1 = _make_span("child-op-1", _TRACE_ID, _SPAN_ID_CHILD, parent=parent_ctx)
    child2 = _make_span("child-op-2", _TRACE_ID, _SPAN_ID_GRANDCHILD, parent=parent_ctx)

    # Export in two separate batches
    exporter.export([child1])
    exporter.export([child2])

    root_spans = [s for s in inner.exported if s.parent is None]
    assert len(root_spans) == 1, "Expected exactly one synthetic root span"


def test_real_root_in_later_batch_does_not_add_duplicate() -> None:
    """If a real root arrives in a second batch, only one synthetic root exists."""
    inner = _CapturingExporter()
    exporter = EnsureRootSpanExporter(inner)

    parent_ctx = _make_span_context(_TRACE_ID, _SPAN_ID_ROOT)
    child = _make_span("child-op", _TRACE_ID, _SPAN_ID_CHILD, parent=parent_ctx)
    real_root = _make_span("root-op", _TRACE_ID, _SPAN_ID_ROOT, parent=None)

    exporter.export([child])   # triggers synthetic root
    exporter.export([real_root])  # real root arrives later; no second synthetic

    root_spans = [s for s in inner.exported if s.parent is None]
    # synthetic root + real root = 2, which is acceptable (Braintrust handles it)
    assert len(root_spans) == 2


def test_multiple_traces_each_get_one_root() -> None:
    """Spans from different traces each receive their own synthetic root."""
    trace_id_2 = _TRACE_ID + 1
    inner = _CapturingExporter()
    exporter = EnsureRootSpanExporter(inner)

    parent_ctx_1 = _make_span_context(_TRACE_ID, _SPAN_ID_ROOT)
    parent_ctx_2 = _make_span_context(trace_id_2, _SPAN_ID_ROOT)
    child1 = _make_span("op1", _TRACE_ID, _SPAN_ID_CHILD, parent=parent_ctx_1)
    child2 = _make_span("op2", trace_id_2, _SPAN_ID_GRANDCHILD, parent=parent_ctx_2)

    exporter.export([child1, child2])

    root_spans = [s for s in inner.exported if s.parent is None]
    root_trace_ids = {s.context.trace_id for s in root_spans if s.context}
    assert root_trace_ids == {_TRACE_ID, trace_id_2}


def test_synthetic_root_inherits_resource_and_timestamps() -> None:
    """The synthetic root shares the child's resource and timestamps."""
    inner = _CapturingExporter()
    exporter = EnsureRootSpanExporter(inner)

    parent_ctx = _make_span_context(_TRACE_ID, _SPAN_ID_ROOT)
    child = _make_span("child-op", _TRACE_ID, _SPAN_ID_CHILD, parent=parent_ctx)
    exporter.export([child])

    synthetic = inner.exported[0]
    assert synthetic.resource is child.resource
    assert synthetic.start_time == child.start_time
    assert synthetic.end_time == child.end_time


def test_delegates_shutdown_and_force_flush() -> None:
    """shutdown() and force_flush() are forwarded to the wrapped exporter."""
    wrapped = MagicMock()
    wrapped.force_flush.return_value = True
    exporter = EnsureRootSpanExporter(wrapped)

    exporter.shutdown()
    wrapped.shutdown.assert_called_once()

    result = exporter.force_flush(timeout_millis=5000)
    wrapped.force_flush.assert_called_once_with(5000)
    assert result is True


def test_export_result_propagated() -> None:
    """The SpanExportResult from the wrapped exporter is returned unchanged."""
    wrapped = MagicMock()
    wrapped.export.return_value = SpanExportResult.FAILURE
    exporter = EnsureRootSpanExporter(wrapped)

    root = _make_span("op", _TRACE_ID, _SPAN_ID_ROOT, parent=None)
    result = exporter.export([root])

    assert result == SpanExportResult.FAILURE
