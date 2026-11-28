"""Input extraction must not break the call being instrumented."""

from collections.abc import Generator
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
from agno.workflow.step import Step
from agno.workflow.types import StepOutput
from agno.workflow.workflow import Workflow
from openinference.semconv.trace import SpanAttributes
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.agno import AgnoInstrumentor
from openinference.instrumentation.agno._workflow_wrapper import (
    _get_input_from_args,
)


@pytest.mark.parametrize(
    "arguments",
    [
        {"step_input": SimpleNamespace(input={"pdf_source": b"\xf2\xad"})},
        {"step_input": SimpleNamespace(get_last_step_content=lambda: {"pdf_source": b"\xf2\xad"})},
        {"execution_input": SimpleNamespace(input={"pdf_source": b"\xf2\xad"})},
    ],
)
def test_bytes_in_input_is_serialized_without_raising(
    arguments: Mapping[str, Any],
) -> None:
    result = _get_input_from_args(arguments)

    assert "pdf_source" in result
    assert "b'" in result


@pytest.fixture()
def span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(span_exporter: InMemorySpanExporter) -> TracerProvider:
    provider = TracerProvider(resource=Resource(attributes={}))
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture()
def instrumented(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    AgnoInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    AgnoInstrumentor().uninstrument()


def _bytes_step(step_input: Any) -> StepOutput:
    payload = step_input.input
    assert isinstance(payload, dict)
    return StepOutput(step_name="bytes_step", content=str(payload["pdf_source"]), success=True)


def test_bytes_input_runs_step_and_is_recorded(
    span_exporter: InMemorySpanExporter,
    instrumented: None,
) -> None:
    workflow = Workflow(
        name="BytesWorkflow",
        steps=[Step(name="bytes_step", executor=_bytes_step)],
    )

    result = workflow.run(input={"pdf_source": b"\xf2\xad"})

    assert result is not None
    assert "b'\\xf2\\xad'" in str(result.content)

    spans = span_exporter.get_finished_spans()
    workflow_span = next(span for span in spans if "BytesWorkflow.run" in span.name)
    step_span = next(span for span in spans if "bytes_step" in span.name)
    for span in (workflow_span, step_span):
        value = dict(span.attributes or {}).get(SpanAttributes.INPUT_VALUE)
        assert value is not None
        assert "pdf_source" in str(value)
        assert "xf2" in str(value)
        assert "xad" in str(value)
