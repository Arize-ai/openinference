"""Test embedding functionality for BeeAI instrumentation."""

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from beeai_framework import BeeAI
from beeai_framework.backend import EmbeddingModel
from beeai_framework.backend.events import (
    EmbeddingModelStartEvent,
    EmbeddingModelSuccessEvent,
)
from beeai_framework.context import RunContext
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.beeai import BeeAIInstrumentor
from openinference.semconv.trace import (
    EmbeddingAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


@pytest.fixture
def in_memory_span_exporter():
    """Create an in-memory span exporter for testing."""
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(in_memory_span_exporter):
    """Create a tracer provider with in-memory exporter."""
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument_beeai(tracer_provider):
    """Automatically instrument BeeAI for each test."""
    BeeAIInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    BeeAIInstrumentor().uninstrument()


def test_embedding_single_text(in_memory_span_exporter):
    """Test embedding span creation with single text input."""
    # Create mock embedding model
    embedding_model = MagicMock(spec=EmbeddingModel)
    embedding_model.model_id = "test-embedding-model"
    embedding_model.provider_id = "test-provider"
    
    # Create mock context
    run_context = MagicMock(spec=RunContext)
    run_context.instance = embedding_model
    
    # Create mock events
    start_event = MagicMock(spec=EmbeddingModelStartEvent)
    start_event.input = MagicMock()
    start_event.input.values = ["Hello, world!"]
    
    success_event = MagicMock(spec=EmbeddingModelSuccessEvent)
    success_event.value = MagicMock()
    success_event.value.embeddings = [[0.1, 0.2, 0.3]]
    success_event.value.usage = MagicMock()
    success_event.value.usage.total_tokens = 10
    success_event.value.usage.prompt_tokens = 8
    success_event.value.usage.completion_tokens = 2
    
    # Simulate the embedding process
    with patch("beeai_framework.context.RunContext") as mock_context:
        mock_context.return_value = run_context
        
        # Import and use the processor
        from openinference.instrumentation.beeai.processors.embedding import (
            EmbeddingModelProcessor,
        )
        
        # Create processor with mock event and meta
        meta = MagicMock()
        meta.creator = run_context
        meta.name = "embedding_start"
        meta.path = "test.path"
        meta.created_at = 0
        
        processor = EmbeddingModelProcessor(start_event, meta)
        
        # Process events
        asyncio.run(processor.update(start_event, meta))
        
        meta.name = "embedding_success"
        asyncio.run(processor.update(success_event, meta))
        
        # Finish the span
        processor.finish()
    
    # Check the exported spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) > 0
    
    # Find the embedding span
    embedding_span = next(
        (s for s in spans if s.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.EMBEDDING.value),
        None
    )
    assert embedding_span is not None
    
    # Verify span attributes
    assert embedding_span.name == "CreateEmbeddings"
    assert embedding_span.attributes.get(SpanAttributes.EMBEDDING_MODEL_NAME) == "test-embedding-model"
    assert embedding_span.attributes.get(SpanAttributes.LLM_PROVIDER) == "test-provider"
    assert embedding_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "beeai"
    
    # Verify embedding text attributes
    text_attr = f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}"
    assert embedding_span.attributes.get(text_attr) == "Hello, world!"
    
    # Verify embedding vector attributes
    vector_attr = f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}"
    assert embedding_span.attributes.get(vector_attr) == [0.1, 0.2, 0.3]
    
    # Verify token counts
    assert embedding_span.attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 10
    assert embedding_span.attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 8
    assert embedding_span.attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 2


def test_embedding_multiple_texts(in_memory_span_exporter):
    """Test embedding span creation with multiple text inputs."""
    # Create mock embedding model
    embedding_model = MagicMock(spec=EmbeddingModel)
    embedding_model.model_id = "test-embedding-model-multi"
    embedding_model.provider_id = "test-provider"
    
    # Create mock context
    run_context = MagicMock(spec=RunContext)
    run_context.instance = embedding_model
    
    # Create mock events with multiple texts
    start_event = MagicMock(spec=EmbeddingModelStartEvent)
    start_event.input = MagicMock()
    start_event.input.values = ["First text", "Second text", "Third text"]
    
    success_event = MagicMock(spec=EmbeddingModelSuccessEvent)
    success_event.value = MagicMock()
    success_event.value.embeddings = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6]
    ]
    success_event.value.usage = None  # Test without usage data
    
    # Simulate the embedding process
    with patch("beeai_framework.context.RunContext") as mock_context:
        mock_context.return_value = run_context
        
        from openinference.instrumentation.beeai.processors.embedding import (
            EmbeddingModelProcessor,
        )
        
        meta = MagicMock()
        meta.creator = run_context
        meta.name = "embedding_start"
        meta.path = "test.path"
        meta.created_at = 0
        
        processor = EmbeddingModelProcessor(start_event, meta)
        
        asyncio.run(processor.update(start_event, meta))
        
        meta.name = "embedding_success"
        asyncio.run(processor.update(success_event, meta))
        
        processor.finish()
    
    spans = in_memory_span_exporter.get_finished_spans()
    
    embedding_span = next(
        (s for s in spans if s.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.EMBEDDING.value),
        None
    )
    assert embedding_span is not None
    
    # Verify all text inputs are recorded
    for i, text in enumerate(["First text", "Second text", "Third text"]):
        text_attr = f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{i}.{EmbeddingAttributes.EMBEDDING_TEXT}"
        assert embedding_span.attributes.get(text_attr) == text
    
    # Verify all vectors are recorded
    expected_vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    for i, vector in enumerate(expected_vectors):
        vector_attr = f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{i}.{EmbeddingAttributes.EMBEDDING_VECTOR}"
        assert embedding_span.attributes.get(vector_attr) == vector


def test_embedding_invocation_parameters(in_memory_span_exporter):
    """Test that embedding invocation parameters are correctly recorded."""
    embedding_model = MagicMock(spec=EmbeddingModel)
    embedding_model.model_id = "test-model-with-params"
    embedding_model.provider_id = "test-provider"
    
    run_context = MagicMock(spec=RunContext)
    run_context.instance = embedding_model
    
    # Create event with additional parameters
    start_event = MagicMock(spec=EmbeddingModelStartEvent)
    start_event.input = MagicMock()
    start_event.input.values = ["Test text"]
    # Add additional attributes that should be captured as invocation parameters
    start_event.input.dimension = 768
    start_event.input.normalize = True
    
    success_event = MagicMock(spec=EmbeddingModelSuccessEvent)
    success_event.value = MagicMock()
    success_event.value.embeddings = [[0.1] * 768]  # Match dimension
    success_event.value.usage = None
    
    with patch("beeai_framework.context.RunContext") as mock_context:
        mock_context.return_value = run_context
        
        from openinference.instrumentation.beeai.processors.embedding import (
            EmbeddingModelProcessor,
        )
        
        meta = MagicMock()
        meta.creator = run_context
        meta.name = "embedding_start"
        meta.path = "test.path"
        meta.created_at = 0
        
        processor = EmbeddingModelProcessor(start_event, meta)
        
        asyncio.run(processor.update(start_event, meta))
        
        meta.name = "embedding_success"
        asyncio.run(processor.update(success_event, meta))
        
        processor.finish()
    
    spans = in_memory_span_exporter.get_finished_spans()
    
    embedding_span = next(
        (s for s in spans if s.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.EMBEDDING.value),
        None
    )
    assert embedding_span is not None
    
    # Verify invocation parameters are captured
    invocation_params = embedding_span.attributes.get(SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS)
    assert invocation_params is not None
    # Should contain dimension and normalize but not values
    assert "dimension" in invocation_params
    assert "normalize" in invocation_params
    assert "values" not in invocation_params