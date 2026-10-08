import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
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
