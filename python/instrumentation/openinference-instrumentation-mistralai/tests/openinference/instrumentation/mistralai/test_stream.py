from types import SimpleNamespace
from typing import Any, AsyncIterator, List

import pytest
from opentelemetry import trace as trace_api

from openinference.instrumentation.mistralai._stream import _AsyncStream
from openinference.instrumentation.mistralai._with_span import _WithSpan


class _FakeAsyncEventStream:
    """A minimal async-iterable standing in for the Mistral SDK's stream,
    which `_AsyncStream.stream_async_with_accumulator` awaits before
    iterating (`async for event in await self.stream`)."""

    def __init__(self, events: List[Any]) -> None:
        self._events = events

    def __await__(self) -> Any:
        async def _self() -> "_FakeAsyncEventStream":
            return self

        return _self().__await__()

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Any]:
        for event in self._events:
            yield event


def _make_with_span(tracer_provider: trace_api.TracerProvider) -> _WithSpan:
    tracer = tracer_provider.get_tracer(__name__)
    span = tracer.start_span("test-span")
    return _WithSpan(span=span)


@pytest.mark.asyncio
async def test_async_stream_tolerates_empty_choices_chunk(
    tracer_provider: trace_api.TracerProvider,
) -> None:
    """Regression: a chunk with an empty `choices` list (e.g. a trailing
    usage-only chunk) used to raise IndexError from inside the generator,
    propagating into the caller's `async for` instead of being swallowed —
    exactly what this repo's instrumentation contract forbids. Every other
    `choices` access in this package already guards against an empty list;
    this one didn't.
    """
    usage_only_chunk = SimpleNamespace(
        data=SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=10))
    )
    normal_chunk = SimpleNamespace(
        data=SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop")],
        )
    )

    with_span = _make_with_span(tracer_provider)
    async_stream = _AsyncStream(
        stream=_FakeAsyncEventStream([normal_chunk, usage_only_chunk]),
        with_span=with_span,
    )

    received = []
    async for event in await async_stream.stream_async_with_accumulator():
        received.append(event)

    assert received == [normal_chunk, usage_only_chunk]
