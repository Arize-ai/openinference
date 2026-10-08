from itertools import product
from typing import Iterator

import pytest
from llama_index.core.instrumentation import get_dispatcher  # type: ignore[attr-defined]
from llama_index.core.instrumentation.events.embedding import SparseEmbeddingEndEvent
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.semconv.trace import (
    EmbeddingAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

dispatcher = get_dispatcher(__name__)


@dispatcher.span  # type: ignore[misc,unused-ignore]
def foo(m: int, n: int) -> None:
    for i in range(m):
        chunks = [f"{i}-{j}" for j in range(n)]
        embeddings = [{j: float(i + j), j + 1: float(i * j)} for j in range(n)]
        dispatcher.event(SparseEmbeddingEndEvent(chunks=chunks, embeddings=embeddings))


async def test_multiple_sparse_embedding_events(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    m, n = 3, 2
    foo(m, n)
    span = in_memory_span_exporter.get_finished_spans()[0]
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND, None) == OpenInferenceSpanKindValues.CHAIN.value
    assert attributes.pop(INPUT_VALUE, None) is not None
    assert attributes.pop(INPUT_MIME_TYPE, None) == OpenInferenceMimeTypeValues.JSON.value

    for k, (i, j) in enumerate(product(range(m), range(n))):
        text = f"{i}-{j}"
        vector = {j: float(i + j), j + 1: float(i * j)}
        assert attributes.pop(f"{EMBEDDING_EMBEDDINGS}.{k}.{EMBEDDING_TEXT}", None) == text
        # Sparse vectors are stored as JSON strings
        import json

        stored = attributes.pop(f"{EMBEDDING_EMBEDDINGS}.{k}.{EMBEDDING_VECTOR}", None)
        assert isinstance(stored, str)
        assert json.loads(stored) == {str(k): v for k, v in vector.items()}

    assert attributes == {}


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[None]:
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    LlamaIndexInstrumentor().uninstrument()


OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
EMBEDDING_EMBEDDINGS = SpanAttributes.EMBEDDING_EMBEDDINGS
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
