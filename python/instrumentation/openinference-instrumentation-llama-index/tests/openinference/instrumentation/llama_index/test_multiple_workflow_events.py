from typing import Iterator

import pytest
from llama_index.core.instrumentation import get_dispatcher  # type: ignore[attr-defined]
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from workflows.runtime.types.step_function import (
    SpanCancelledEvent,
    WorkflowRunOutputEvent,
    WorkflowStepOutputEvent,
)

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.semconv.trace import SpanAttributes

dispatcher = get_dispatcher(__name__)


@dispatcher.span  # type: ignore[misc,unused-ignore]
def step_span() -> None:
    dispatcher.event(WorkflowStepOutputEvent(output="StopEvent(result='hello')"))


@dispatcher.span  # type: ignore[misc,unused-ignore]
def run_span() -> None:
    dispatcher.event(WorkflowRunOutputEvent(output="StopEvent(result='done')"))


@dispatcher.span  # type: ignore[misc,unused-ignore]
def cancelled_span() -> None:
    dispatcher.event(SpanCancelledEvent(reason="workflow cancelled by user"))


async def test_workflow_step_output_event(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    step_span()
    span = in_memory_span_exporter.get_finished_spans()[0]
    assert span.attributes
    assert span.attributes[OUTPUT_VALUE] == "StopEvent(result='hello')"


async def test_workflow_run_output_event(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    run_span()
    span = in_memory_span_exporter.get_finished_spans()[0]
    assert span.attributes
    assert span.attributes[OUTPUT_VALUE] == "StopEvent(result='done')"


async def test_span_cancelled_event_sets_otel_span_event(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    cancelled_span()
    span = in_memory_span_exporter.get_finished_spans()[0]
    from opentelemetry.trace import StatusCode

    assert span.status.status_code != StatusCode.ERROR
    matching = [e for e in span.events if e.name == "span.cancelled"]
    assert matching, f"Expected a 'span.cancelled' span event, got: {[e.name for e in span.events]}"
    assert matching[0].attributes
    assert matching[0].attributes["reason"] == "workflow cancelled by user"


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[None]:
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    LlamaIndexInstrumentor().uninstrument()


OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
