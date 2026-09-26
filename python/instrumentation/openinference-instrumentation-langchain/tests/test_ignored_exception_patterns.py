import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.tracers.schemas import Run
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.langchain._tracer import OpenInferenceTracer, _update_span


def _make_run(error: str) -> Run:
    now = datetime.now(timezone.utc)
    return Run(
        id=uuid.uuid4(),
        name="graph",
        start_time=now,
        end_time=now,
        run_type="chain",
        error=error,
        inputs={},
        outputs={},
        extra={},
    )


@pytest.mark.parametrize(
    "error",
    [
        "Command(goto='next')",
        "ParentCommand(graph='parent')",
        "GraphInterrupt((Interrupt(value='confirm?'),))",
    ],
)
def test_ignored_control_flow_exceptions_set_span_status_ok(error: str) -> None:
    """Ignored control-flow exceptions set the span status to OK."""
    span = MagicMock()

    _update_span(span, _make_run(error))

    span.set_status.assert_called_once_with(trace_api.StatusCode.OK)


@pytest.mark.parametrize(
    "error",
    [
        "KeyError('missing key')",
        "RuntimeError('runtime error')",
        "TypeError('invalid type')",
        "ValueError('invalid value')",
    ],
)
def test_other_exceptions_set_span_status_error(error: str) -> None:
    """Other exceptions set the span status to ERROR."""
    span = MagicMock()

    _update_span(span, _make_run(error))

    (status,), _ = span.set_status.call_args
    assert status.status_code is trace_api.StatusCode.ERROR


class _Command:
    """Stands in for `langgraph.types.Command`, which is not a test dependency.

    Only its `repr` matters: the control-flow filter matches on `repr(error)`.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return f"Command({', '.join(f'{k}={v!r}' for k, v in self._kwargs.items())})"


class ParentCommand(Exception):
    """Stands in for `langgraph.errors.ParentCommand`."""


class GraphInterrupt(Exception):
    """Stands in for `langgraph.errors.GraphInterrupt`."""


def _run_chain_error(error: BaseException) -> ReadableSpan:
    """Drive a chain span through the real error callback and return the finished span."""
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OpenInferenceTracer(
        tracer=tracer_provider.get_tracer(__name__),
        separate_trace_from_runtime_context=False,
    )

    run_id = uuid.uuid4()
    tracer.on_chain_start({"name": "graph"}, {"messages": []}, run_id=run_id)
    try:
        raise error
    except BaseException as e:
        tracer.on_chain_error(e, run_id=run_id)

    (span,) = exporter.get_finished_spans()
    return span


@pytest.mark.parametrize(
    "error",
    [
        ParentCommand(_Command(goto="collector", update={"ticket_id": "Alert-477638"})),
        GraphInterrupt(("Interrupt(value='confirm?')",)),
    ],
)
def test_control_flow_exceptions_record_no_exception_event(error: BaseException) -> None:
    """Control-flow exceptions are not real failures, so they leave no exception event.

    Span status alone is not enough: observability UIs flag any span carrying an
    `exception` event, so a span reported OK while holding one is contradictory.
    """
    span = _run_chain_error(error)

    assert [event.name for event in span.events] == []
    assert span.status.status_code is trace_api.StatusCode.OK


def test_genuine_exceptions_are_still_recorded_as_events() -> None:
    """Filtering control-flow signals must not suppress real failures."""
    span = _run_chain_error(ValueError("invalid value"))

    (event,) = span.events
    assert event.name == "exception"
    assert event.attributes is not None
    assert event.attributes["exception.type"] == "ValueError"
    assert event.attributes["exception.message"] == "invalid value"
    assert span.status.status_code is trace_api.StatusCode.ERROR


def test_exception_named_like_a_control_flow_signal_is_still_recorded() -> None:
    """The filter anchors on the exception's own repr, not on text inside its message."""
    span = _run_chain_error(ValueError("Command(goto='collector') was rejected"))

    assert [event.name for event in span.events] == ["exception"]
    assert span.status.status_code is trace_api.StatusCode.ERROR
