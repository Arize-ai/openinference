from typing import Any, Dict, Iterator

import litellm
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode, TracerProvider

from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.semconv.trace import EmbeddingAttributes, SpanAttributes

# TODO: Update to use SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS when released in semconv
_EMBEDDING_INVOCATION_PARAMETERS = "embedding.invocation_parameters"


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
) -> Iterator[None]:
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    yield


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_batch_embedding(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    input_texts = ["hello", "world", "test"]

    response = litellm.embedding(
        model="openai/text-embedding-ada-002",
        api_key="sk-",
        input=input_texts,
    )

    assert response is not None
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "CreateEmbeddings"
    assert span.status.status_code == StatusCode.OK

    attributes: Dict[str, Any] = dict(span.attributes or {})
    assert attributes is not None

    # Check model name
    assert attributes.pop(SpanAttributes.EMBEDDING_MODEL_NAME) == "text-embedding-ada-002-v2"

    # Check embedding texts
    for i, text in enumerate(input_texts):
        assert (
            attributes.pop(
                f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{i}.{EmbeddingAttributes.EMBEDDING_TEXT}"
            )
            == text
        )

    # Check embedding vectors
    for i in range(len(input_texts)):
        vector = attributes.pop(
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{i}.{EmbeddingAttributes.EMBEDDING_VECTOR}"
        )
        assert vector is not None
        assert isinstance(vector, tuple)
        assert len(vector) > 0

    # Check token counts
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 3
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 3

    # Check span kind
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "EMBEDDING"

    # Check input value
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == "['hello', 'world', 'test']"

    # Check invocation parameters (api_key should be redacted for security)
    assert (
        attributes.pop(_EMBEDDING_INVOCATION_PARAMETERS)
        == '{"model": "openai/text-embedding-ada-002"}'
    )

    # All attributes should be accounted for
    assert attributes == {}


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_single_string_embedding(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    input_text = "hello world"

    response = litellm.embedding(
        model="openai/text-embedding-ada-002",
        api_key="sk-",
        input=input_text,
    )

    assert response is not None
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "CreateEmbeddings"
    assert span.status.status_code == StatusCode.OK

    attributes: Dict[str, Any] = dict(span.attributes or {})
    assert attributes is not None

    # Check model name
    assert attributes.pop(SpanAttributes.EMBEDDING_MODEL_NAME) == "text-embedding-ada-002-v2"

    # Check single embedding text
    assert (
        attributes.pop(
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}"
        )
        == input_text
    )

    # Check single embedding vector
    vector = attributes.pop(
        f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}"
    )
    assert vector is not None
    assert isinstance(vector, tuple)
    assert len(vector) > 0

    # Check token counts
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 2
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 2

    # Check span kind
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "EMBEDDING"

    # Check input value
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == "hello world"

    # Check invocation parameters (api_key should be redacted for security)
    assert (
        attributes.pop(_EMBEDDING_INVOCATION_PARAMETERS)
        == '{"model": "openai/text-embedding-ada-002"}'
    )

    # All attributes should be accounted for
    assert attributes == {}


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_batch_embedding_with_different_model(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Test embeddings with text-embedding-3-small model."""
    input_texts = ["first text", "second text"]

    response = litellm.embedding(
        model="openai/text-embedding-3-small",
        api_key="sk-",
        input=input_texts,
    )

    assert response is not None
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "CreateEmbeddings"
    assert span.status.status_code == StatusCode.OK

    attributes: Dict[str, Any] = dict(span.attributes or {})
    assert attributes is not None

    # Check model name
    assert attributes.pop(SpanAttributes.EMBEDDING_MODEL_NAME) == "text-embedding-3-small"

    # Check embedding texts
    for i, text in enumerate(input_texts):
        assert (
            attributes.pop(
                f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{i}.{EmbeddingAttributes.EMBEDDING_TEXT}"
            )
            == text
        )

    # Check embedding vectors
    for i in range(len(input_texts)):
        vector = attributes.pop(
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{i}.{EmbeddingAttributes.EMBEDDING_VECTOR}"
        )
        assert vector is not None
        assert isinstance(vector, tuple)
        assert len(vector) > 0

    # Check token counts
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 4
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 4

    # Check span kind
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "EMBEDDING"

    # Check input value
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == "['first text', 'second text']"

    # Check invocation parameters (api_key should be redacted for security)
    assert (
        attributes.pop(_EMBEDDING_INVOCATION_PARAMETERS)
        == '{"model": "openai/text-embedding-3-small"}'
    )

    # All attributes should be accounted for
    assert attributes == {}
