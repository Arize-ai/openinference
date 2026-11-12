from types import SimpleNamespace
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, cast

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.agno._model_wrapper import _ModelWrapper


@pytest.mark.parametrize("method", ["run", "arun", "run_stream", "arun_stream"])
@pytest.mark.parametrize("fails", [False, True], ids=["success", "error"])
async def test_model_span_status_reflects_invocation_outcome(method: str, fails: bool) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    wrapper = _ModelWrapper(provider.get_tracer(__name__))
    model = SimpleNamespace(name="Probe", id="gpt-4o-mini", provider="OpenAI")
    response = SimpleNamespace(
        role="assistant", content="Hello", tool_calls=None, reasoning_content=None
    )

    def run() -> Any:
        if fails:
            raise RuntimeError("model request failed")
        return response

    async def arun() -> Any:
        if fails:
            raise RuntimeError("model request failed")
        return response

    def run_stream() -> Iterator[Any]:
        yield response
        if fails:
            raise RuntimeError("model request failed")

    async def arun_stream() -> AsyncIterator[Any]:
        yield response
        if fails:
            raise RuntimeError("model request failed")

    async def invoke() -> None:
        if method == "run":
            assert wrapper.run(run, model, (), {}) is response
        elif method == "arun":
            assert await wrapper.arun(arun, model, (), {}) is response
        elif method == "run_stream":
            assert list(wrapper.run_stream(run_stream, model, (), {})) == [response]
        else:
            assert [
                chunk
                async for chunk in wrapper.arun_stream(
                    cast(Callable[..., Awaitable[Any]], arun_stream), model, (), {}
                )
            ] == [response]

    if fails:
        with pytest.raises(RuntimeError, match="model request failed"):
            await invoke()
    else:
        await invoke()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert dict(span.attributes or {})["openinference.span.kind"] == "LLM"
    if fails:
        assert span.status.status_code == trace_api.StatusCode.ERROR
        assert [event.name for event in span.events] == ["exception"]
    else:
        assert span.status.status_code == trace_api.StatusCode.OK
        assert "output.value" in dict(span.attributes or {})
