from typing import Any

import pytest
from opentelemetry import context as context_api
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


def test_default_activation_delegates_to_global_context() -> None:
    activation = SpanActivation()
    global_baseline = trace_api.get_current_span()
    span = trace_api.NonRecordingSpan(trace_api.SpanContext(1, 1, is_remote=False))

    token = activation.attach_span(span)
    try:
        assert trace_api.get_current_span() is span
    finally:
        activation.detach(token)

    assert trace_api.get_current_span() is global_baseline


def test_injected_runtime_context_never_touches_global_context() -> None:
    remote_span = trace_api.NonRecordingSpan(trace_api.SpanContext(1, 1, is_remote=False))
    global_token = context_api.attach(trace_api.set_span_in_context(remote_span))
    try:
        activation = SpanActivation(ContextVarsRuntimeContext())
        agno_span = trace_api.NonRecordingSpan(trace_api.SpanContext(2, 2, is_remote=False))

        token = activation.attach_span(agno_span)
        try:
            assert activation.get_current_span() is agno_span
            assert trace_api.get_current_span() is remote_span
        finally:
            activation.detach(token)

        assert trace_api.get_current_span() is remote_span
    finally:
        context_api.detach(global_token)


def test_attach_value_builds_from_activation_current_not_global() -> None:
    activation = SpanActivation(ContextVarsRuntimeContext())

    outer_token = activation.attach_value("outer", "outer-value")
    try:
        inner_token = activation.attach_value("inner", "inner-value")
        try:
            assert activation.get_value("outer") == "outer-value"
            assert activation.get_value("inner") == "inner-value"
            assert context_api.get_value("outer") is None
            assert context_api.get_value("inner") is None
        finally:
            activation.detach(inner_token)
    finally:
        activation.detach(outer_token)


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
