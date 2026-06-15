import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

from langchain_core.tracers.schemas import Run
from opentelemetry import trace as trace_api

from openinference.instrumentation.langchain._tracer import _update_span


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


def test_graph_interrupt_sets_span_status_ok() -> None:
    """GraphInterrupt is expected LangGraph control flow, not an error."""
    span = MagicMock()
    _update_span(span, _make_run("GraphInterrupt((Interrupt(value='confirm?'),))"))
    span.set_status.assert_called_once_with(trace_api.StatusCode.OK)


def test_real_exception_sets_span_status_error() -> None:
    """A genuine exception still marks the span as an error."""
    span = MagicMock()
    _update_span(span, _make_run("ValueError('boom')"))
    (status,), _ = span.set_status.call_args
    assert status.status_code is trace_api.StatusCode.ERROR
