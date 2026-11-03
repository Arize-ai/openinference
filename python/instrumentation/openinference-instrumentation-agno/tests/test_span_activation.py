from typing import Any

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.context.contextvars_context import ContextVarsRuntimeContext
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.agno._context import SpanActivation


def _tracer(exporter: InMemorySpanExporter) -> Any:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OITracer(
        trace_api.get_tracer("test-span-activation", tracer_provider=tracer_provider),
        config=TraceConfig(),
    )


def test_use_span_ends_on_exit_and_detaches() -> None:
    exporter = InMemorySpanExporter()
    tracer = _tracer(exporter)

    activation = SpanActivation(ContextVarsRuntimeContext())
    span = tracer.start_span("span")

    with activation.use_span(span, end_on_exit=True):
        assert activation.get_current_span() is span

    assert activation.get_current_span() is trace_api.INVALID_SPAN
    (finished,) = exporter.get_finished_spans()
    assert finished.end_time is not None
    assert finished.status.status_code == trace_api.StatusCode.UNSET


def test_use_span_records_exception_and_still_detaches() -> None:
    exporter = InMemorySpanExporter()
    tracer = _tracer(exporter)
    activation = SpanActivation(ContextVarsRuntimeContext())
    span = tracer.start_span("span")

    with pytest.raises(ValueError, match="boom"):
        with activation.use_span(span, end_on_exit=True):
            raise ValueError("boom")

    assert activation.get_current_span() is trace_api.INVALID_SPAN
    (finished,) = exporter.get_finished_spans()
    assert finished.status.status_code == trace_api.StatusCode.ERROR
    assert any(event.name == "exception" for event in finished.events)
