"""A failing instrumented workflow must restore the caller's span."""

from types import SimpleNamespace

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider

from openinference.instrumentation.agno._workflow_wrapper import _WorkflowWrapper


def test_workflow_run_exception_restores_caller_span() -> None:
    tracer = TracerProvider().get_tracer("test")
    wrapper = _WorkflowWrapper(tracer)
    workflow = SimpleNamespace(name="repro", steps=[])

    def failing() -> None:
        raise ValueError("boom")

    parent = tracer.start_span("parent")
    with trace_api.use_span(parent, end_on_exit=False):
        with pytest.raises(ValueError, match="boom"):
            wrapper.run(failing, workflow, (), {})

        assert trace_api.get_current_span() is parent
    parent.end()
