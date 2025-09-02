"""Test batch embedding functionality for LiteLLM instrumentation."""

from typing import Any, cast
from unittest.mock import patch

import litellm
from litellm import OpenAIChatCompletion  # type: ignore[attr-defined]
from litellm.types.utils import EmbeddingResponse, Usage
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from openinference.semconv.trace import EmbeddingAttributes, SpanAttributes


def test_batch_embedding(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Test that batch embeddings (multiple inputs) are properly instrumented."""
    in_memory_span_exporter.clear()

    # Mock response with multiple embeddings matching the input
    mock_response_embedding = EmbeddingResponse(
        model="text-embedding-ada-002",
        data=[
            {"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"},
            {"embedding": [0.4, 0.5, 0.6], "index": 1, "object": "embedding"},
            {"embedding": [0.7, 0.8, 0.9], "index": 2, "object": "embedding"},
        ],
        object="list",
        usage=Usage(prompt_tokens=18, completion_tokens=3, total_tokens=21),
    )

    input_texts = ["hello", "world", "test"]

    with patch.object(OpenAIChatCompletion, "embedding", return_value=mock_response_embedding):
        litellm.embedding(model="text-embedding-ada-002", input=input_texts)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "CreateEmbeddings"
    attributes = dict(cast(Any, span.attributes))

    # Check model name
    assert attributes.get(SpanAttributes.EMBEDDING_MODEL_NAME) == "text-embedding-ada-002"

    # Check each input text is recorded
    for i, text in enumerate(input_texts):
        assert (
            attributes.get(
                f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{i}.{EmbeddingAttributes.EMBEDDING_TEXT}"
            )
            == text
        )

    # Check each output vector is recorded
    expected_vectors = [
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6),
        (0.7, 0.8, 0.9),
    ]
    for i, vector in enumerate(expected_vectors):
        assert (
            attributes.get(
                f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{i}.{EmbeddingAttributes.EMBEDDING_VECTOR}"
            )
            == vector
        )

    # Check token counts
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 18
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 3
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 21

    assert span.status.status_code == StatusCode.OK


def test_single_string_embedding(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Test that single string input (not in list) is properly instrumented."""
    in_memory_span_exporter.clear()

    mock_response_embedding = EmbeddingResponse(
        model="text-embedding-ada-002",
        data=[
            {"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"},
        ],
        object="list",
        usage=Usage(prompt_tokens=6, completion_tokens=1, total_tokens=7),
    )

    with patch.object(OpenAIChatCompletion, "embedding", return_value=mock_response_embedding):
        litellm.embedding(model="text-embedding-ada-002", input="hello world")

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(cast(Any, span.attributes))

    # Single string should still be recorded at index 0
    assert (
        attributes.get(
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}"
        )
        == "hello world"
    )
    assert attributes.get(
        f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}"
    ) == (0.1, 0.2, 0.3)


def test_token_ids_embedding_no_text_attributes(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Test that token IDs (integers) as input do NOT produce text attributes."""
    in_memory_span_exporter.clear()

    mock_response_embedding = EmbeddingResponse(
        model="text-embedding-ada-002",
        data=[
            {"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"},
        ],
        object="list",
        usage=Usage(prompt_tokens=3, completion_tokens=1, total_tokens=4),
    )

    # Input as token IDs (integers) instead of text
    token_ids = [15339, 1917, 123]  # Example token IDs

    with patch.object(OpenAIChatCompletion, "embedding", return_value=mock_response_embedding):
        litellm.embedding(model="text-embedding-ada-002", input=token_ids)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(cast(Any, span.attributes))

    # Token IDs should NOT produce text attributes
    assert (
        f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}"
        not in attributes
    ), "Token IDs should not produce text attributes"

    # But vectors should still be recorded
    assert attributes.get(
        f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}"
    ) == (0.1, 0.2, 0.3)

    # Model name and token counts should still be present
    assert attributes.get(SpanAttributes.EMBEDDING_MODEL_NAME) == "text-embedding-ada-002"
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 3
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 4
