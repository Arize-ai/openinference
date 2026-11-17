"""A failing instrumented workflow, step, or parallel run must restore the caller's span."""

from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider

from openinference.instrumentation.agno._workflow_wrapper import (
    _ParallelWrapper,
    _StepWrapper,
    _WorkflowWrapper,
)


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


def test_step_run_exception_restores_caller_span() -> None:
    tracer = TracerProvider().get_tracer("test")
    wrapper = _StepWrapper(tracer)
    step = SimpleNamespace(name="repro_step")

    def failing(step_input: Any) -> None:
        raise ValueError("boom")

    parent = tracer.start_span("parent")
    with trace_api.use_span(parent, end_on_exit=False):
        with pytest.raises(ValueError, match="boom"):
            wrapper.run(failing, step, (SimpleNamespace(input="hello"),), {})

        assert trace_api.get_current_span() is parent
    parent.end()


def test_parallel_execute_exception_restores_caller_span() -> None:
    tracer = TracerProvider().get_tracer("test")
    wrapper = _ParallelWrapper(tracer)
    parallel = SimpleNamespace(name="repro_parallel", steps=[])

    def failing(step_input: Any) -> None:
        raise ValueError("boom")

    parent = tracer.start_span("parent")
    with trace_api.use_span(parent, end_on_exit=False):
        with pytest.raises(ValueError, match="boom"):
            wrapper.execute(failing, parallel, (SimpleNamespace(input="hello"),), {})

        assert trace_api.get_current_span() is parent
    parent.end()
