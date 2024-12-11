from asyncio import gather, sleep
from random import random
from typing import Iterator

import pytest
from llama_index.core.instrumentation import get_dispatcher
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor, get_current_span

dispatcher = get_dispatcher(__name__)


@dispatcher.span
async def foo(k: int) -> str:
    if k > 1:
        await gather(sleep(random() / 100), foo(k - 1))
    if (span := get_current_span()) is None:
        return ""
    return str(span.get_span_context().span_id)


async def test_get_current_span(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    n, k = 100, 3
    await gather(*(foo(k) for _ in range(n)))
    assert len(spans := in_memory_span_exporter.get_finished_spans()) == n * k
    seen = set()
    for span in spans:
        assert span.attributes and span.context
        assert (expected := str(span.context.span_id)) not in seen
        seen.add(expected)
        assert span.attributes.get("output.value") == expected


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[None]:
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    LlamaIndexInstrumentor().uninstrument()
