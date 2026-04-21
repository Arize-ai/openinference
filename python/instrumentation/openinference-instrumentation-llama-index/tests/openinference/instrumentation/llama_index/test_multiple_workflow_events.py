import asyncio
from typing import Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


class IntermediateEvent(Event):
    value: int


class TwoStepWorkflow(Workflow):
    @step
    async def first(self, ev: StartEvent) -> IntermediateEvent:
        return IntermediateEvent(value=ev.get("value", 0) * 2)

    @step
    async def second(self, ev: IntermediateEvent) -> StopEvent:
        return StopEvent(result=ev.value + 1)


class SlowWorkflow(Workflow):
    @step
    async def slow(self, ev: StartEvent) -> StopEvent:
        _step_started.set()
        await asyncio.sleep(30)
        return StopEvent(result="never")


_step_started: asyncio.Event


async def test_workflow_step_output_event(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    handler = TwoStepWorkflow().run(value=5)
    await handler
    spans = {s.name: s for s in in_memory_span_exporter.get_finished_spans()}

    step_span = spans["TwoStepWorkflow.first"]
    step_attributes = dict(step_span.attributes or {})
    assert (
        step_attributes.pop(OPENINFERENCE_SPAN_KIND, None)
        == OpenInferenceSpanKindValues.CHAIN.value
    )
    assert step_attributes.pop(INPUT_VALUE, None) is not None
    assert step_attributes.pop(INPUT_MIME_TYPE, None) == OpenInferenceMimeTypeValues.JSON.value
    assert step_attributes.pop(OUTPUT_VALUE, None) == "IntermediateEvent(value=10)"
    assert step_attributes.pop(OUTPUT_MIME_TYPE, None) == OpenInferenceMimeTypeValues.TEXT.value
    assert step_attributes == {}


async def test_workflow_run_output_event(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    handler = TwoStepWorkflow().run(value=5)
    await handler
    spans = {s.name: s for s in in_memory_span_exporter.get_finished_spans()}

    run_span = spans["TwoStepWorkflow.run"]
    run_attributes = dict(run_span.attributes or {})
    assert (
        run_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == OpenInferenceSpanKindValues.CHAIN.value
    )
    assert run_attributes.pop(INPUT_VALUE, None) is not None
    assert run_attributes.pop(INPUT_MIME_TYPE, None) == OpenInferenceMimeTypeValues.JSON.value
    assert run_attributes.pop(OUTPUT_VALUE, None) == "StopEvent(result=11)"
    assert run_attributes.pop(OUTPUT_MIME_TYPE, None) == OpenInferenceMimeTypeValues.TEXT.value
    assert run_attributes == {}


async def test_span_cancelled_event(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    global _step_started
    _step_started = asyncio.Event()

    handler = SlowWorkflow(timeout=30).run()
    await _step_started.wait()
    await handler.cancel_run()

    spans = {s.name: s for s in in_memory_span_exporter.get_finished_spans()}
    run_span = spans["SlowWorkflow.run"]

    assert run_span.status.status_code != StatusCode.ERROR
    cancelled_events = [e for e in run_span.events if e.name == "span.cancelled"]
    assert cancelled_events, (
        f"Expected 'span.cancelled' event on run span, got: {[e.name for e in run_span.events]}"
    )
    assert cancelled_events[0].attributes
    assert cancelled_events[0].attributes["reason"] == "workflow cancelled by user"


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[None]:
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    LlamaIndexInstrumentor().uninstrument()


OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
