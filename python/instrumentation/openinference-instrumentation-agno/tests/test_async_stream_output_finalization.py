import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable, cast

import pytest
from agno.agent import Agent
from agno.run.agent import RunCompletedEvent, RunOutput
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.agno._runs_wrapper import _RunWrapper


class FinalAnswer(BaseModel):
    answer: str


async def _fake_arun_stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[RunOutput]:
    yield RunOutput(
        run_id="run-123",
        content=FinalAnswer(answer="done"),
    )


async def _fake_arun_stream_events(*_args: Any, **_kwargs: Any) -> AsyncIterator[object]:
    yield RunCompletedEvent(content=FinalAnswer(answer="done"))
    yield RunOutput(
        run_id="run-123",
        content=FinalAnswer(answer="done"),
    )


def _build_wrapper() -> tuple[_RunWrapper, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OITracer(
        trace_api.get_tracer("test-agno-arun-stream", tracer_provider=tracer_provider),
        config=TraceConfig(),
    )
    return _RunWrapper(tracer=tracer), exporter  # type: ignore[arg-type]


FAKE_ARUN_STREAM = cast(Callable[..., Awaitable[Any]], _fake_arun_stream)
FAKE_ARUN_STREAM_EVENTS = cast(Callable[..., Awaitable[Any]], _fake_arun_stream_events)


@pytest.mark.asyncio
@pytest.mark.parametrize("close_mode", ["aclose", "break"])
async def test_arun_stream_records_output_value_when_closed_after_final_run_output(
    close_mode: str,
) -> None:
    wrapper, exporter = _build_wrapper()
    agent = Agent(name="test-agent")

    if close_mode == "aclose":
        stream = wrapper.arun_stream(
            FAKE_ARUN_STREAM,
            None,
            (agent,),
            {"yield_run_output": True},
        )
        event = await anext(stream)
        assert isinstance(event, RunOutput)
        await stream.aclose()
        await asyncio.sleep(0)
    else:
        async for event in wrapper.arun_stream(
            FAKE_ARUN_STREAM,
            None,
            (agent,),
            {"yield_run_output": True},
        ):
            if isinstance(event, RunOutput):
                break
        await asyncio.sleep(0.01)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    assert attributes.get("output.value") == '{"answer":"done"}'


@pytest.mark.asyncio
async def test_arun_stream_records_output_value_when_stream_events_close_on_run_completed() -> None:
    wrapper, exporter = _build_wrapper()
    agent = Agent(name="test-agent")

    async for event in wrapper.arun_stream(
        FAKE_ARUN_STREAM_EVENTS,
        None,
        (agent,),
        {"yield_run_output": True, "stream_events": True},
    ):
        if isinstance(event, RunCompletedEvent):
            break

    await asyncio.sleep(0.05)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    assert attributes.get("output.value") == '{"answer":"done"}'
