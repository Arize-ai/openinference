from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, cast

import pytest
from agno.agent import Agent
from agno.run import RunStatus
from agno.run.agent import RunOutput
from agno.run.team import TeamRunOutput
from agno.team import Team
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.agno._runs_wrapper import _RunWrapper


@pytest.fixture
def wrapper_and_exporter() -> tuple[_RunWrapper, InMemorySpanExporter, TracerProvider]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return _RunWrapper(provider.get_tracer(__name__)), exporter, provider


@pytest.mark.parametrize("method", ["run", "arun", "run_stream", "arun_stream"])
@pytest.mark.parametrize("run_type", ["agent", "team"])
@pytest.mark.parametrize("run_status", [RunStatus.completed, RunStatus.error])
async def test_run_span_uses_returned_status(
    wrapper_and_exporter: tuple[_RunWrapper, InMemorySpanExporter, TracerProvider],
    method: str,
    run_type: str,
    run_status: RunStatus,
) -> None:
    wrapper, exporter, _ = wrapper_and_exporter
    actor = Agent(name="Probe") if run_type == "agent" else Team(name="Probe", members=[])
    output_type = RunOutput if run_type == "agent" else TeamRunOutput
    output = output_type(run_id="run-1", content="result", status=run_status)

    def run(_actor: Any) -> Any:
        return output

    async def arun(_actor: Any) -> Any:
        return output

    def run_stream(_actor: Any, **_kwargs: Any) -> Iterator[Any]:
        yield output

    async def arun_stream(_actor: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        yield output

    if method == "run":
        assert wrapper.run(run, None, (actor,), {}) is output
    elif method == "arun":
        assert await wrapper.arun(arun, None, (actor,), {}) is output
    elif method == "run_stream":
        assert list(wrapper.run_stream(run_stream, None, (actor,), {})) == []
    else:
        assert [
            item
            async for item in wrapper.arun_stream(
                cast(Callable[..., Awaitable[Any]], arun_stream), None, (actor,), {}
            )
        ] == []

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert dict(spans[0].attributes or {})["output.value"] == "result"
    expected_status = (
        trace_api.StatusCode.ERROR if run_status == RunStatus.error else trace_api.StatusCode.OK
    )
    assert spans[0].status.status_code == expected_status
    if run_status == RunStatus.error:
        assert spans[0].status.description == "run status: error"


@pytest.mark.parametrize("method", ["run_stream", "arun_stream"])
async def test_stream_without_run_output_remains_ok(
    wrapper_and_exporter: tuple[_RunWrapper, InMemorySpanExporter, TracerProvider],
    method: str,
) -> None:
    wrapper, exporter, _ = wrapper_and_exporter
    agent = Agent(name="Probe")

    def run_stream(_actor: Any, **_kwargs: Any) -> Iterator[Any]:
        yield "event"

    async def arun_stream(_actor: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        yield "event"

    if method == "run_stream":
        assert list(wrapper.run_stream(run_stream, None, (agent,), {})) == ["event"]
    else:
        assert [
            item
            async for item in wrapper.arun_stream(
                cast(Callable[..., Awaitable[Any]], arun_stream), None, (agent,), {}
            )
        ] == ["event"]

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == trace_api.StatusCode.OK


def test_failed_child_does_not_change_successful_run_status(
    wrapper_and_exporter: tuple[_RunWrapper, InMemorySpanExporter, TracerProvider],
) -> None:
    wrapper, exporter, provider = wrapper_and_exporter
    agent = Agent(name="Probe")
    tracer = provider.get_tracer(__name__)

    def run(_actor: Agent) -> RunOutput:
        with tracer.start_as_current_span("failed-attempt") as child:
            child.set_status(trace_api.StatusCode.ERROR)
        return RunOutput(run_id="run-1", status=RunStatus.completed)

    wrapper.run(run, None, (agent,), {})

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["failed-attempt"].status.status_code == trace_api.StatusCode.ERROR
    assert spans["Probe.run"].status.status_code == trace_api.StatusCode.OK
