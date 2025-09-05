"""Test embedding functionality for BeeAI instrumentation."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from beeai_framework.backend import EmbeddingModel
from beeai_framework.backend.events import (
    EmbeddingModelStartEvent,
    EmbeddingModelSuccessEvent,
)
from beeai_framework.context import RunContext
from openinference.semconv.trace import EmbeddingAttributes, SpanAttributes

from openinference.instrumentation.beeai.processors.embedding import (
    EmbeddingModelProcessor,
    _EMBEDDING_INVOCATION_PARAMETERS,
)


def test_embedding_invocation_parameters_constant():
    """Test that the local EMBEDDING_INVOCATION_PARAMETERS constant is properly defined."""
    assert _EMBEDDING_INVOCATION_PARAMETERS == "embedding.invocation_parameters"


def test_embedding_processor_sets_attributes():
    """Test that EmbeddingModelProcessor sets the correct attributes."""
    # Create mock embedding model
    embedding_model = MagicMock(spec=EmbeddingModel)
    embedding_model.model_id = "test-model"
    embedding_model.provider_id = "test-provider"

    # Create mock context
    run_context = MagicMock(spec=RunContext)
    run_context.instance = embedding_model

    # Create mock meta
    meta = MagicMock()
    meta.creator = run_context
    meta.trace = MagicMock()
    meta.trace.run_id = "test-run-id"
    meta.name = "embedding_start"
    meta.path = "test.path"
    meta.created_at = 0

    # Create mock start event with invocation parameters
    start_event = MagicMock(spec=EmbeddingModelStartEvent)
    start_event.input = MagicMock()
    start_event.input.values = ["test text"]
    start_event.input.dimension = 768
    start_event.input.normalize = True

    # Create processor
    processor = EmbeddingModelProcessor(start_event, meta)

    # Verify span was created with correct attributes
    assert processor.span.name == "CreateEmbeddings"
    assert processor.span.attributes[SpanAttributes.EMBEDDING_MODEL_NAME] == "test-model"
    assert processor.span.attributes[SpanAttributes.LLM_PROVIDER] == "test-provider"
    assert processor.span.attributes[SpanAttributes.LLM_SYSTEM] == "beeai"

    # Test that update method handles EmbeddingModelStartEvent
    asyncio.run(processor.update(start_event, meta))

    # Check embedding text was set
    text_attr = f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}"
    assert processor.span.attributes.get(text_attr) == "test text"

    # Check invocation parameters were set (excluding 'values')
    invocation_params_str = processor.span.attributes.get(_EMBEDDING_INVOCATION_PARAMETERS)
    assert invocation_params_str is not None
    invocation_params = json.loads(invocation_params_str)
    assert invocation_params["dimension"] == 768
    assert invocation_params["normalize"] is True
    assert "values" not in invocation_params

    # Create mock success event
    success_event = MagicMock(spec=EmbeddingModelSuccessEvent)
    success_event.value = MagicMock()
    success_event.value.embeddings = [[0.1, 0.2, 0.3]]
    success_event.value.usage = MagicMock()
    success_event.value.usage.total_tokens = 10
    success_event.value.usage.prompt_tokens = 8
    success_event.value.usage.completion_tokens = 2

    # Test that update method handles EmbeddingModelSuccessEvent
    asyncio.run(processor.update(success_event, meta))

    # Check embedding vector was set
    vector_attr = f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}"
    assert processor.span.attributes.get(vector_attr) == [0.1, 0.2, 0.3]

    # Check token usage was set
    assert processor.span.attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 10
    assert processor.span.attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 8
    assert processor.span.attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 2
