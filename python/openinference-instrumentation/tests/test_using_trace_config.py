import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import (
    REDACTED_VALUE,
    OITracer,
    TraceConfig,
    get_current_trace_config,
    using_trace_config,
)
from openinference.semconv.trace import SpanAttributes

HIDE_ALL = dict(hide_inputs=True, hide_outputs=True)
BASE = TraceConfig()


def _emit(tracer: OITracer, name: str = "span") -> None:
    with tracer.start_as_current_span(name, openinference_span_kind="chain") as span:
        span.set_input("secret-in")
        span.set_output("secret-out")


def _attrs(exporter: InMemorySpanExporter) -> Dict[str, Any]:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return dict(spans[0].attributes or {})


def test_default_config_used_without_override(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    assert get_current_trace_config(BASE) is BASE
    _emit(tracer)
    attrs = _attrs(in_memory_span_exporter)
    assert attrs[SpanAttributes.INPUT_VALUE] == "secret-in"
    assert attrs[SpanAttributes.OUTPUT_VALUE] == "secret-out"


def test_override_applies_inside_block_only(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    with using_trace_config(**HIDE_ALL):
        assert get_current_trace_config(BASE).hide_inputs
        _emit(tracer)
    attrs = _attrs(in_memory_span_exporter)
    assert attrs[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert attrs[SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE

    in_memory_span_exporter.clear()
    assert get_current_trace_config(BASE) is BASE
    _emit(tracer)
    assert _attrs(in_memory_span_exporter)[SpanAttributes.INPUT_VALUE] == "secret-in"


def test_nesting_restores_outer_config(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    with using_trace_config(hide_inputs=True):
        with using_trace_config(hide_outputs=True):
            _emit(tracer, "inner")
        _emit(tracer, "outer")
    spans = {s.name: dict(s.attributes or {}) for s in in_memory_span_exporter.get_finished_spans()}
    # The inner block merges on top of the outer one.
    assert spans["inner"][SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert spans["inner"][SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE
    assert spans["outer"][SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert spans["outer"][SpanAttributes.OUTPUT_VALUE] == "secret-out"


def test_decorator_sync(tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter) -> None:
    @using_trace_config(**HIDE_ALL)
    def fn() -> None:
        _emit(tracer)

    fn()
    assert _attrs(in_memory_span_exporter)[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert get_current_trace_config(BASE) is BASE


async def test_async_context_manager_and_decorator(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    async with using_trace_config(**HIDE_ALL):
        await asyncio.sleep(0)
        _emit(tracer, "cm")

    @using_trace_config(**HIDE_ALL)
    async def fn() -> None:
        await asyncio.sleep(0)
        _emit(tracer, "deco")

    await fn()
    spans = {s.name: dict(s.attributes or {}) for s in in_memory_span_exporter.get_finished_spans()}
    for name in ("cm", "deco"):
        assert spans[name][SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
        assert spans[name][SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE
    assert get_current_trace_config(BASE) is BASE


async def test_concurrent_tasks_are_isolated(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    async def hidden() -> None:
        with using_trace_config(**HIDE_ALL):
            await asyncio.sleep(0)
            _emit(tracer, "hidden")

    async def visible() -> None:
        await asyncio.sleep(0)
        _emit(tracer, "visible")

    await asyncio.gather(hidden(), visible())
    spans = {s.name: dict(s.attributes or {}) for s in in_memory_span_exporter.get_finished_spans()}
    assert spans["hidden"][SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert spans["visible"][SpanAttributes.INPUT_VALUE] == "secret-in"


def test_span_ended_outside_block_keeps_override(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    with using_trace_config(**HIDE_ALL):
        span = tracer.start_span("stream", openinference_span_kind="llm")
        span.set_input("secret-in")
    # Simulates a streamed response finishing after the block exits.
    span.set_output("secret-out")
    span.end()
    attrs = _attrs(in_memory_span_exporter)
    assert attrs[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert attrs[SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE


def test_span_started_outside_block_ignores_override(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    span = tracer.start_span("pre", openinference_span_kind="chain")
    with using_trace_config(**HIDE_ALL):
        span.set_input("secret-in")
    span.end()
    assert _attrs(in_memory_span_exporter)[SpanAttributes.INPUT_VALUE] == "secret-in"


def test_override_merges_with_instrumentor_config() -> None:
    base = TraceConfig(hide_outputs=True, base64_image_max_length=10)
    with using_trace_config(hide_inputs=True):
        merged = get_current_trace_config(base)
    assert merged.hide_inputs is True
    assert merged.hide_outputs is True
    assert merged.base64_image_max_length == 10
    assert base.hide_inputs is False


@pytest.mark.parametrize("trace_config", [TraceConfig(hide_inputs=True)])
def test_override_can_unhide_instrumentor_setting(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    with using_trace_config(hide_inputs=False):
        _emit(tracer)
    assert _attrs(in_memory_span_exporter)[SpanAttributes.INPUT_VALUE] == "secret-in"


def test_override_keeps_instrumentor_blob_uploader() -> None:
    class _Uploader:
        def upload(self, blob: Any) -> None:
            return None

        def shutdown(self, timeout_sec: float = 0) -> None:
            pass

    uploader = _Uploader()
    base = TraceConfig(blob_uploader=uploader)
    with using_trace_config(hide_inputs=True):
        assert get_current_trace_config(base).blob_uploader is uploader
    with using_trace_config(blob_uploader=None):
        assert get_current_trace_config(base).blob_uploader is None


@pytest.mark.parametrize(
    "overrides",
    [{"hide_everything": True}, {"hide_inputs": "yes"}, {"base64_image_max_length": "1"}],
)
def test_invalid_overrides_rejected(overrides: Dict[str, Any]) -> None:
    with pytest.raises(TypeError):
        using_trace_config(**overrides)


def test_thread_pool_workers_need_context_propagation(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    with using_trace_config(**HIDE_ALL), ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_emit, tracer, "plain").result()
        pool.submit(contextvars.copy_context().run, _emit, tracer, "propagated").result()
    spans = {s.name: dict(s.attributes or {}) for s in in_memory_span_exporter.get_finished_spans()}
    assert spans["plain"][SpanAttributes.INPUT_VALUE] == "secret-in"
    assert spans["propagated"][SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
