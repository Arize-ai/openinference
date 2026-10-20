from types import SimpleNamespace
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, cast

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import suppress_tracing
from openinference.instrumentation.agno._model_wrapper import _ModelWrapper
from openinference.instrumentation.agno._runs_wrapper import _RunWrapper


def _wrappers() -> tuple[InMemorySpanExporter, _ModelWrapper, _RunWrapper]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    return exporter, _ModelWrapper(tracer), _RunWrapper(tracer)


def test_model_run_stream_passes_chunks_through_when_tracing_is_suppressed() -> None:
    exporter, model_wrapper, _ = _wrappers()
    model = SimpleNamespace(name="Probe", id="gpt-4o-mini", provider="OpenAI")

    def run_stream() -> Iterator[Any]:
        yield "a"
        yield "b"

    with suppress_tracing():
        assert list(model_wrapper.run_stream(run_stream, model, (), {})) == ["a", "b"]
    assert exporter.get_finished_spans() == ()


async def test_model_arun_returns_the_response_when_tracing_is_suppressed() -> None:
    exporter, model_wrapper, _ = _wrappers()
    model = SimpleNamespace(name="Probe", id="gpt-4o-mini", provider="OpenAI")

    async def arun() -> Any:
        return "response"

    with suppress_tracing():
        assert await model_wrapper.arun(arun, model, (), {}) == "response"
    assert exporter.get_finished_spans() == ()


def test_run_stream_passes_events_through_when_tracing_is_suppressed() -> None:
    exporter, _, run_wrapper = _wrappers()
    agent = SimpleNamespace(name="Probe")

    def run_stream() -> Iterator[Any]:
        yield "a"
        yield "b"

    with suppress_tracing():
        assert list(run_wrapper.run_stream(run_stream, agent, (), {})) == ["a", "b"]
    assert exporter.get_finished_spans() == ()


async def test_arun_stream_passes_events_through_when_tracing_is_suppressed() -> None:
    exporter, model_wrapper, run_wrapper = _wrappers()
    model = SimpleNamespace(name="Probe", id="gpt-4o-mini", provider="OpenAI")
    agent = SimpleNamespace(name="Probe")

    async def model_stream() -> AsyncIterator[Any]:
        yield "a"

    async def run_stream() -> AsyncIterator[Any]:
        yield "a"

    async def arun_stream_coro() -> AsyncIterator[Any]:
        return run_stream()

    with suppress_tracing():
        assert [
            e
            async for e in model_wrapper.arun_stream(
                cast(Callable[..., Awaitable[Any]], model_stream), model, (), {}
            )
        ] == ["a"]
        assert [
            e
            async for e in run_wrapper.arun_stream(
                cast(Callable[..., Awaitable[Any]], arun_stream_coro), agent, (), {}
            )
        ] == ["a"]
    assert exporter.get_finished_spans() == ()
