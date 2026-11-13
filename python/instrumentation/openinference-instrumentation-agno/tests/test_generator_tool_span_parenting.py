"""Spans opened inside a generator tool's body must be children of the tool span."""

from typing import Any, AsyncIterator, Iterator

import pytest
from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.agno._tools_wrapper import _FunctionCallWrapper
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def _build_wrapper() -> tuple[_FunctionCallWrapper, OITracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OITracer(
        trace_api.get_tracer("test-agno-generator-tool", tracer_provider=tracer_provider),
        config=TraceConfig(),
    )
    return _FunctionCallWrapper(tracer=tracer), tracer, exporter  # type: ignore[arg-type]


def _child_parent_span_id(exporter: InMemorySpanExporter) -> int | None:
    spans = {span.name: span for span in exporter.get_finished_spans()}
    child = spans["child"]
    assert child.parent is not None, "child span has no parent"
    return child.parent.span_id


def test_sync_generator_tool_body_spans_are_children_of_the_tool_span() -> None:
    wrapper, tracer, exporter = _build_wrapper()
    tool_span = tracer.start_span("tool")

    def body() -> Iterator[str]:
        with tracer.start_as_current_span("child"):
            yield "value"

    assert list(wrapper._handle_success_streaming(body(), tool_span)) == ["value"]

    assert _child_parent_span_id(exporter) == tool_span.context.span_id


@pytest.mark.asyncio
async def test_async_generator_tool_body_spans_are_children_of_the_tool_span() -> None:
    wrapper, tracer, exporter = _build_wrapper()
    tool_span = tracer.start_span("tool")

    async def body() -> AsyncIterator[str]:
        with tracer.start_as_current_span("child"):
            yield "value"

    assert [item async for item in wrapper._handle_async_success_streaming(body(), tool_span)] == [
        "value"
    ]

    assert _child_parent_span_id(exporter) == tool_span.context.span_id


def test_sync_generator_tool_span_is_still_ended() -> None:
    wrapper, tracer, exporter = _build_wrapper()
    tool_span = tracer.start_span("tool")

    def body() -> Iterator[Any]:
        yield "value"

    list(wrapper._handle_success_streaming(body(), tool_span))

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert "tool" in spans
