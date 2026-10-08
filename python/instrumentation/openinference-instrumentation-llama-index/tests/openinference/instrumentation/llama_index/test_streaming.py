import asyncio
from contextvars import ContextVar
from typing import Any, AsyncGenerator, Generator, cast

import pytest
from llama_index.core.base.response.schema import AsyncStreamingResponse, StreamingResponse
from llama_index.core.instrumentation import get_dispatcher  # type: ignore[attr-defined]
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

dispatcher = get_dispatcher(__name__)


@pytest.mark.parametrize("separate_trace", [False, True])
@pytest.mark.parametrize("completion", ["exhaust", "error", "close", "cancel", "throw"])
def test_async_response_context(
    completion: str,
    separate_trace: bool,
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    LlamaIndexInstrumentor().instrument(
        tracer_provider=tracer_provider, separate_trace_from_runtime_context=separate_trace
    )
    state: ContextVar[str] = ContextVar("producer_state", default="consumer")

    async def run() -> None:
        entered = asyncio.Event()

        @dispatcher.span
        async def child() -> None:
            await asyncio.sleep(0)
            if completion == "cancel":
                entered.set()
                await asyncio.Event().wait()
            if completion == "error":
                raise ValueError("stream failed")

        async def tokens(label: str) -> AsyncGenerator[str, None]:
            token = state.set(label)
            try:
                await asyncio.sleep(0)
                yield label
                assert state.get() == label
                await child()
                yield "!"
            finally:
                await asyncio.sleep(0)
                assert state.get() == label
                state.reset(token)

        @dispatcher.span
        def query(label: str) -> AsyncStreamingResponse:
            return AsyncStreamingResponse(response_gen=tokens(label))

        streams = [query(label).response_gen for label in ("first", "second")]
        for label, stream in zip(("first", "second"), streams):
            assert await stream.__anext__() == label
            assert state.get() == "consumer"
            assert not trace_api.get_current_span().get_span_context().is_valid
        with tracer_provider.get_tracer(__name__).start_as_current_span("unrelated"):
            pass
        for stream in streams:
            if completion == "close":
                await stream.aclose()
            elif completion == "cancel":
                entered.clear()

                async def consume() -> str:
                    return await stream.__anext__()

                task = asyncio.create_task(consume())
                await entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            elif completion in ("error", "throw"):
                with pytest.raises(ValueError, match="stream failed"):
                    if completion == "throw":
                        await stream.athrow(ValueError("stream failed"))
                    else:
                        await stream.__anext__()
            else:
                assert await stream.asend(None) == "!"
                assert [token async for token in stream] == []
            await stream.aclose()
            with pytest.raises(StopAsyncIteration):
                await stream.__anext__()
            assert state.get() == "consumer"
            assert not trace_api.get_current_span().get_span_context().is_valid

    asyncio.run(run())
    spans = in_memory_span_exporter.get_finished_spans()
    queries = [span for span in spans if span.name.endswith("query")]
    children = [span for span in spans if span.name.endswith("child")]
    assert len(queries) == 2
    assert len(children) == (0 if completion in ("close", "throw") else 2)
    assert len(spans) == len(queries) + len(children) + 1
    assert len({span.context.trace_id for span in queries}) == 2
    assert next(span for span in spans if span.name == "unrelated").parent is None
    for query in queries:
        assert query.status.status_code == (
            trace_api.StatusCode.ERROR
            if completion in ("error", "cancel", "throw")
            else trace_api.StatusCode.OK
        )
    for child in children:
        query = next(span for span in queries if span.context.trace_id == child.context.trace_id)
        assert child.parent is not None
        assert child.parent.span_id == query.context.span_id
        assert query.start_time is not None and query.end_time is not None
        assert child.start_time is not None and child.end_time is not None
        assert query.start_time <= child.start_time <= child.end_time <= query.end_time


@pytest.mark.parametrize("completion", ["exhaust", "error", "close", "throw"])
def test_sync_response_context(
    completion: str,
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    state: ContextVar[str] = ContextVar("producer_state", default="consumer")

    @dispatcher.span
    def child() -> None:
        if completion == "error":
            raise ValueError("stream failed")

    def tokens(label: str) -> Generator[str, None, None]:
        token = state.set(label)
        try:
            yield label
            assert state.get() == label
            child()
            yield "!"
        finally:
            assert state.get() == label
            state.reset(token)

    @dispatcher.span
    def query(label: str) -> StreamingResponse:
        return StreamingResponse(response_gen=tokens(label))

    streams = [cast(Any, query(label).response_gen) for label in ("first", "second")]
    for label, stream in zip(("first", "second"), streams):
        assert next(stream) == label
        assert state.get() == "consumer"
        assert not trace_api.get_current_span().get_span_context().is_valid
    for stream in streams:
        if completion == "close":
            stream.close()
        elif completion in ("error", "throw"):
            with pytest.raises(ValueError, match="stream failed"):
                if completion == "throw":
                    stream.throw(ValueError("stream failed"))
                else:
                    next(stream)
        else:
            assert stream.send(None) == "!"
            assert list(stream) == []
        stream.close()
        assert state.get() == "consumer"
        assert not trace_api.get_current_span().get_span_context().is_valid

    spans = in_memory_span_exporter.get_finished_spans()
    queries = [span for span in spans if span.name.endswith("query")]
    children = [span for span in spans if span.name.endswith("child")]
    assert len(queries) == 2
    assert len(children) == (0 if completion in ("close", "throw") else 2)
    assert len(spans) == len(queries) + len(children)
    assert len({span.context.trace_id for span in queries}) == 2
    for query_span in queries:
        assert query_span.status.status_code == (
            trace_api.StatusCode.ERROR
            if completion in ("error", "throw")
            else trace_api.StatusCode.OK
        )
    for child_span in children:
        query_span = next(
            span for span in queries if span.context.trace_id == child_span.context.trace_id
        )
        assert child_span.parent is not None
        assert child_span.parent.span_id == query_span.context.span_id
