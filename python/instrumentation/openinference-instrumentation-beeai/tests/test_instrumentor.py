import json
import os
from typing import Mapping, cast

import pytest
from beeai_framework.adapters.openai import OpenAIEmbeddingModel
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue

from openinference.semconv.trace import (
    EmbeddingAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
@pytest.mark.asyncio
async def test_openai_embeddings(in_memory_span_exporter: InMemorySpanExporter) -> None:
    """Test that BeeAI OpenAI embeddings are properly traced."""
    # API key from environment - only used when re-recording the cassette
    # When using the cassette, the key is not needed
    api_key = os.getenv("OPENAI_API_KEY", "sk-test")

    # Create an embedding model
    embedding_model = OpenAIEmbeddingModel(
        model_id="text-embedding-3-small",
        api_key=api_key,
    )

    # Create embeddings for test texts
    texts = ["Hello world", "Test embedding"]

    # Run the embedding request
    response = await embedding_model.create(texts)

    # Verify we got embeddings back
    assert response is not None
    assert response.embeddings is not None
    assert len(response.embeddings) == 2

    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1

    # Get the single span
    openinference_span = spans[0]
    assert openinference_span is not None

    # Verify span attributes
    attributes = dict(cast(Mapping[str, AttributeValue], openinference_span.attributes))

    # Check basic attributes as per spec
    assert (
        attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.EMBEDDING.value
    )
    assert attributes.get(SpanAttributes.EMBEDDING_MODEL_NAME) == "text-embedding-3-small"
    assert attributes.get(SpanAttributes.LLM_SYSTEM) == "beeai"
    assert attributes.get(SpanAttributes.LLM_PROVIDER) == "openai"

    # Check embedding texts
    assert (
        attributes.get(
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}"
        )
        == "Hello world"
    )
    assert (
        attributes.get(
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.1.{EmbeddingAttributes.EMBEDDING_TEXT}"
        )
        == "Test embedding"
    )

    # Check embedding vectors exist and have correct structure
    vector_0 = attributes.get(
        f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}"
    )
    vector_1 = attributes.get(
        f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.1.{EmbeddingAttributes.EMBEDDING_VECTOR}"
    )

    assert vector_0 is not None
    assert vector_1 is not None
    # Vectors are tuples in the cassette, check exact length from recorded data
    assert isinstance(vector_0, (list, tuple))
    assert isinstance(vector_1, (list, tuple))
    assert len(vector_0) == 1536  # text-embedding-3-small dimension
    assert len(vector_1) == 1536  # text-embedding-3-small dimension
    # Check first few values are correct floats from cassette
    assert vector_0[0] == pytest.approx(-0.002078542485833168)
    assert vector_0[1] == pytest.approx(-0.04908587411046028)
    assert vector_1[0] == pytest.approx(-0.005330947693437338)
    assert vector_1[1] == pytest.approx(-0.03916504979133606)

    # Check invocation parameters
    invocation_params = attributes.get("embedding.invocation_parameters")
    assert isinstance(invocation_params, str)
    assert json.loads(invocation_params) == {"abort_signal": None, "max_retries": 0}

    # Check token counts
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 4
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 4
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 0
