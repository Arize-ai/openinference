from types import SimpleNamespace
from typing import Any, Iterator, cast

import pytest
from agno.agent import Agent
from openinference.semconv.trace import SpanAttributes
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context.contextvars_context import ContextVarsRuntimeContext
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import TraceConfig
from openinference.instrumentation.agno import AgnoInstrumentor
from openinference.instrumentation.agno._context import (
    ActivationTracer,
    SpanActivation,
    get_activation,
    set_activation,
)
from openinference.instrumentation.agno._runs_wrapper import _RunWrapper
from openinference.instrumentation.agno._workflow_wrapper import _StepWrapper, _WorkflowWrapper


@pytest.fixture()
def isolated_runtime_context() -> Iterator[ContextVarsRuntimeContext]:
    runtime_context = ContextVarsRuntimeContext()
    set_activation(SpanActivation(runtime_context))
    try:
        yield runtime_context
    finally:
        set_activation(SpanActivation())


def _tracer(exporter: InMemorySpanExporter) -> Any:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return ActivationTracer(
        trace_api.get_tracer("test-runtime-context-isolation", tracer_provider=tracer_provider),
        config=TraceConfig(),
    )


def test_instrument_routes_activation_and_uninstrument_restores_default() -> None:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    runtime_context = ContextVarsRuntimeContext()
    instrumentor = AgnoInstrumentor()

    base_span = trace_api.get_current_span()

    instrumentor.instrument(tracer_provider=tracer_provider, runtime_context=runtime_context)
    try:
        span = trace_api.NonRecordingSpan(trace_api.SpanContext(1, 1, is_remote=False))
        token = get_activation().attach_span(span)
        try:
            assert trace_api.get_current_span(runtime_context.get_current()) is span
            assert trace_api.get_current_span() is base_span
        finally:
            get_activation().detach(token)
    finally:
        instrumentor.uninstrument()

    span = trace_api.NonRecordingSpan(trace_api.SpanContext(2, 2, is_remote=False))
    token = get_activation().attach_span(span)
    try:
        assert trace_api.get_current_span() is span
    finally:
        get_activation().detach(token)


def test_run_wrapper_activates_only_in_isolated_context(
    isolated_runtime_context: ContextVarsRuntimeContext,
) -> None:
    exporter = InMemorySpanExporter()
    wrapper = _RunWrapper(tracer=_tracer(exporter))

    agent = Agent(name="test-agent")
    base_span = trace_api.get_current_span()
    observed = {}

    def fake_run(*_args, **_kwargs) -> Any:
        observed["isolated_has_span"] = trace_api.get_current_span(
            isolated_runtime_context.get_current()
        ).is_recording()
        observed["global_untouched"] = trace_api.get_current_span() is base_span
        return SimpleNamespace(run_id="run-1", content="done")

    wrapper.run(fake_run, None, (agent,), {})

    assert observed == {"isolated_has_span": True, "global_untouched": True}
    assert trace_api.get_current_span() is base_span
    assert len(exporter.get_finished_spans()) == 1


def test_run_stream_wrapper_activates_only_in_isolated_context(
    isolated_runtime_context: ContextVarsRuntimeContext,
) -> None:
    exporter = InMemorySpanExporter()
    wrapper = _RunWrapper(tracer=_tracer(exporter))

    agent = Agent(name="test-agent")
    base_span = trace_api.get_current_span()

    def fake_run_stream(*_args, **_kwargs) -> Iterator[Any]:
        yield SimpleNamespace(run_id="run-1")

    stream = wrapper.run_stream(fake_run_stream, None, (agent,), {})
    next(stream)  # enters the generator body: activation is now attached

    assert trace_api.get_current_span(isolated_runtime_context.get_current()).is_recording()
    assert trace_api.get_current_span() is base_span

    with pytest.raises(StopIteration):
        next(stream)  # drains the generator: activation is detached, span ends

    assert trace_api.get_current_span() is base_span
    assert len(exporter.get_finished_spans()) == 1


class _StepInstance:
    name = "test-step"
    agent = None
    team = None


class _WorkflowInstance:
    name = "test-workflow"
    description = None
    steps = [_StepInstance()]
    id = "workflow-1"
    user_id = None


def test_workflow_and_step_activate_only_in_isolated_context_and_keep_hierarchy(
    isolated_runtime_context: ContextVarsRuntimeContext,
) -> None:
    exporter = InMemorySpanExporter()
    tracer = _tracer(exporter)
    workflow_wrapper = _WorkflowWrapper(tracer=tracer)
    step_wrapper = _StepWrapper(tracer=tracer)
    base_span = trace_api.get_current_span()

    def fake_step_execute(*_args, **_kwargs) -> str:
        assert trace_api.get_current_span(isolated_runtime_context.get_current()).is_recording()
        assert trace_api.get_current_span() is base_span
        return "step output"

    def run(*_args, **_kwargs) -> str:
        return cast(str, step_wrapper.run(fake_step_execute, _StepInstance(), (), {}))

    workflow_wrapper.run(run, _WorkflowInstance(), ("hello",), {})

    assert trace_api.get_current_span() is base_span

    by_node_name = {
        (span.attributes or {}).get(SpanAttributes.GRAPH_NODE_NAME): span
        for span in exporter.get_finished_spans()
    }
    workflow_span, step_span = by_node_name["test-workflow"], by_node_name["test-step"]
    assert (step_span.attributes or {})[SpanAttributes.GRAPH_NODE_PARENT_ID] == (
        workflow_span.attributes or {}
    )[SpanAttributes.GRAPH_NODE_ID], "step parent needs to be the workflow"


def test_suppress_instrumentation_key_is_unaffected_by_isolated_activation(
    isolated_runtime_context: ContextVarsRuntimeContext,
) -> None:
    """_SUPPRESS_INSTRUMENTATION_KEY must block span creation whatever the activation."""
    exporter = InMemorySpanExporter()
    wrapper = _RunWrapper(tracer=_tracer(exporter))
    agent = Agent(name="test-agent")
    calls = []

    def fake_run(*_args, **_kwargs) -> str:
        calls.append(True)
        return "unrecorded"

    suppress_token = context_api.attach(
        context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, True)
    )
    try:
        result = wrapper.run(fake_run, None, (agent,), {})
    finally:
        context_api.detach(suppress_token)

    assert result == "unrecorded"
    assert calls == [True]
    assert len(exporter.get_finished_spans()) == 0


def test_isolated_spans_do_not_join_the_host_trace(
    isolated_runtime_context: ContextVarsRuntimeContext,
) -> None:
    """The host keeps a span current in the GLOBAL context: agno must not graft onto it."""
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = _tracer(exporter)
    workflow_wrapper = _WorkflowWrapper(tracer=tracer)
    step_wrapper = _StepWrapper(tracer=tracer)

    host_tracer = tracer_provider.get_tracer("host")
    with host_tracer.start_as_current_span("host") as host_span:
        host_context = host_span.get_span_context()

        def run(*_args: Any, **_kwargs: Any) -> str:
            return cast(str, step_wrapper.run(lambda *a, **k: "out", _StepInstance(), (), {}))

        workflow_wrapper.run(run, _WorkflowInstance(), ("hello",), {})

    by_name = {span.name: span for span in exporter.get_finished_spans()}
    workflow_span = next(s for n, s in by_name.items() if "test_workflow" in n)
    step_span = next(s for n, s in by_name.items() if "test_step" in n)

    assert workflow_span.parent is None, "workflow must be a root span, not a child of the host"
    assert workflow_span.context.trace_id != host_context.trace_id, (
        "isolated spans must not land in the host trace"
    )
    assert step_span.parent is not None
    assert step_span.parent.span_id == workflow_span.context.span_id, (
        "step must be a child of the workflow"
    )
