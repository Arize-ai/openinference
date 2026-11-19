"""
Spans started while a generator tool is consumed must be children of the tool span.

Agno's delegate_task_to_member is a generator: FunctionCall.execute()/aexecute() only
creates it, and the member agent runs later while Agno iterates function_call.result.
"""

from typing import Any, AsyncGenerator, AsyncIterator, Dict, Generator, Iterator, Optional, Type

import pytest
from agno.exceptions import RunCancelledException
from agno.tools.function import Function, FunctionCall
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from openinference.instrumentation.agno import AgnoInstrumentor


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def instrument(tracer_provider: TracerProvider) -> Generator[None, None, None]:
    AgnoInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    AgnoInstrumentor().uninstrument()


def _spans_by_name(exporter: InMemorySpanExporter) -> Dict[str, ReadableSpan]:
    return {span.name: span for span in exporter.get_finished_spans()}


def _assert_parent(spans: Dict[str, ReadableSpan], child: str, parent: str) -> None:
    child_span = spans[child]
    assert child_span.parent is not None, f"{child} has no parent"
    assert child_span.parent.span_id == spans[parent].context.span_id, (
        f"{child} is not a child of {parent}"
    )


def test_sync_generator_tool_span_is_parent_of_spans_started_while_consuming(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    instrument: Any,
) -> None:
    tracer = tracer_provider.get_tracer(__name__)

    def generator_tool() -> Iterator[str]:
        with tracer.start_as_current_span("inner_of_generator_tool"):
            yield "chunk"

    with tracer.start_as_current_span("team_run") as team_span:
        call = FunctionCall(function=Function.from_callable(generator_tool), arguments={})
        call.execute()
        assert isinstance(call.result, Iterator)
        assert list(call.result) == ["chunk"]
        # The tool span is no longer current once the generator is exhausted.
        assert trace_api.get_current_span() is team_span

    spans = _spans_by_name(in_memory_span_exporter)
    _assert_parent(spans, "generator_tool", "team_run")
    _assert_parent(spans, "inner_of_generator_tool", "generator_tool")


async def test_async_generator_tool_span_is_parent_of_spans_started_while_consuming(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    instrument: Any,
) -> None:
    tracer = tracer_provider.get_tracer(__name__)

    async def generator_tool() -> AsyncIterator[str]:
        with tracer.start_as_current_span("inner_of_generator_tool"):
            yield "chunk"

    with tracer.start_as_current_span("team_run") as team_span:
        call = FunctionCall(function=Function.from_callable(generator_tool), arguments={})
        await call.aexecute()
        assert isinstance(call.result, AsyncIterator)
        assert [item async for item in call.result] == ["chunk"]
        assert trace_api.get_current_span() is team_span

    spans = _spans_by_name(in_memory_span_exporter)
    _assert_parent(spans, "generator_tool", "team_run")
    _assert_parent(spans, "inner_of_generator_tool", "generator_tool")


_EARLY_STOPS = [
    pytest.param(RuntimeError, StatusCode.ERROR, id="raise"),
    pytest.param(RunCancelledException, StatusCode.UNSET, id="cancel"),
    pytest.param(None, StatusCode.UNSET, id="close"),
]


@pytest.mark.parametrize("error, expected_status", _EARLY_STOPS)
def test_sync_generator_tool_span_ends_when_iteration_stops_early(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    instrument: Any,
    error: Optional[Type[Exception]],
    expected_status: StatusCode,
) -> None:
    tracer = tracer_provider.get_tracer(__name__)

    def generator_tool() -> Iterator[str]:
        with tracer.start_as_current_span("inner_of_generator_tool"):
            yield "chunk"
            if error is not None:
                raise error("tool stopped")

    with tracer.start_as_current_span("team_run") as team_span:
        call = FunctionCall(function=Function.from_callable(generator_tool), arguments={})
        call.execute()
        assert isinstance(call.result, Generator)
        assert next(call.result) == "chunk"
        if error is None:
            call.result.close()
        else:
            with pytest.raises(error, match="tool stopped"):
                next(call.result)
        assert trace_api.get_current_span() is team_span

    spans = _spans_by_name(in_memory_span_exporter)
    assert "generator_tool" in spans, "tool span was not ended"
    _assert_parent(spans, "inner_of_generator_tool", "generator_tool")
    assert spans["generator_tool"].status.status_code is expected_status


@pytest.mark.parametrize("error, expected_status", _EARLY_STOPS)
async def test_async_generator_tool_span_ends_when_iteration_stops_early(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    instrument: Any,
    error: Optional[Type[Exception]],
    expected_status: StatusCode,
) -> None:
    tracer = tracer_provider.get_tracer(__name__)

    async def generator_tool() -> AsyncIterator[str]:
        with tracer.start_as_current_span("inner_of_generator_tool"):
            yield "chunk"
            if error is not None:
                raise error("tool stopped")

    with tracer.start_as_current_span("team_run") as team_span:
        call = FunctionCall(function=Function.from_callable(generator_tool), arguments={})
        await call.aexecute()
        assert isinstance(call.result, AsyncGenerator)
        assert await call.result.__anext__() == "chunk"
        if error is None:
            await call.result.aclose()
        else:
            with pytest.raises(error, match="tool stopped"):
                await call.result.__anext__()
        assert trace_api.get_current_span() is team_span

    spans = _spans_by_name(in_memory_span_exporter)
    assert "generator_tool" in spans, "tool span was not ended"
    _assert_parent(spans, "inner_of_generator_tool", "generator_tool")
    assert spans["generator_tool"].status.status_code is expected_status


async def test_coroutine_tool_span_is_still_parent_of_its_spans(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    instrument: Any,
) -> None:
    tracer = tracer_provider.get_tracer(__name__)

    async def coroutine_tool() -> str:
        with tracer.start_as_current_span("inner_of_coroutine_tool"):
            return "done"

    with tracer.start_as_current_span("team_run"):
        call = FunctionCall(function=Function.from_callable(coroutine_tool), arguments={})
        await call.aexecute()

    spans = _spans_by_name(in_memory_span_exporter)
    _assert_parent(spans, "coroutine_tool", "team_run")
    _assert_parent(spans, "inner_of_coroutine_tool", "coroutine_tool")
