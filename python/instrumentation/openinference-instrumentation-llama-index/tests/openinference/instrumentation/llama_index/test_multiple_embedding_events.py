from itertools import product
from typing import Iterator

import pytest
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.embedding import EmbeddingEndEvent
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.semconv.trace import EmbeddingAttributes, SpanAttributes

dispatcher = get_dispatcher(__name__)


@dispatcher.span  # type: ignore[misc,unused-ignore]
def foo(m: int, n: int) -> None:
    for i in range(m):
        chunks = [f"{i}-{j}" for j in range(n)]
        embeddings = [list(map(float, [i, j])) for j in range(n)]
        dispatcher.event(EmbeddingEndEvent(chunks=chunks, embeddings=embeddings))


async def test_multiple_embedding_events(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    m, n = 3, 2
    foo(m, n)
    span = in_memory_span_exporter.get_finished_spans()[0]
    assert span.attributes
    for k, (i, j) in enumerate(product(range(m), range(n))):
        text, vector = f"{i}-{j}", tuple(map(float, [i, j]))
        assert span.attributes[f"{EMBEDDING_EMBEDDINGS}.{k}.{EMBEDDING_TEXT}"] == text
        assert span.attributes[f"{EMBEDDING_EMBEDDINGS}.{k}.{EMBEDDING_VECTOR}"] == vector


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[None]:
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    LlamaIndexInstrumentor().uninstrument()


EMBEDDING_EMBEDDINGS = SpanAttributes.EMBEDDING_EMBEDDINGS
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
