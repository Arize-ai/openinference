import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import SpanKind
import litellm
from openinference.semconv.trace import SpanAttributes, EmbeddingAttributes, OpenInferenceSpanKindValues
from openinference.instrumentation.litellm import LiteLLMInstrumentor
import json

@pytest.fixture(scope="class")
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()

@pytest.fixture(scope="class")
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider

@pytest.fixture(scope="class")
def instrumentor(tracer_provider: TracerProvider) -> LiteLLMInstrumentor:
    instrumentor = LiteLLMInstrumentor(tracer_provider=tracer_provider)
    instrumentor.instrument()
    yield instrumentor
    instrumentor.uninstrument()

class TestLiteLLMInstrumentor:
    def test_completion(tracer_provider, in_memory_span_exporter, instrumentor):
        in_memory_span_exporter.clear()

        litellm.completion(model="gpt-3.5-turbo", messages=[{"content": "What's the capital of China?", "role": "user"}])
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == 'completion'
        assert span.kind == SpanKind.INTERNAL
        assert span.attributes[SpanAttributes.LLM_MODEL_NAME] == 'gpt-3.5-turbo'
        assert span.attributes[SpanAttributes.INPUT_VALUE] == "What's the capital of China?"
        assert SpanAttributes.LLM_INVOCATION_PARAMETERS in span.attributes

    def test_acompletion(tracer_provider, in_memory_span_exporter, instrumentor):
        in_memory_span_exporter.clear()

        import asyncio
        asyncio.run(litellm.acompletion(model="gpt-3.5-turbo", messages=[{"content": "What's the capital of China?", "role": "user"}]))
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == 'acompletion'
        assert span.kind == SpanKind.INTERNAL
        assert span.attributes[SpanAttributes.LLM_MODEL_NAME] == 'gpt-3.5-turbo'
        assert span.attributes[SpanAttributes.INPUT_VALUE] == "What's the capital of China?"
        assert SpanAttributes.LLM_INVOCATION_PARAMETERS in span.attributes

    def test_completion_with_retries(tracer_provider, in_memory_span_exporter, instrumentor):
        in_memory_span_exporter.clear()

        litellm.completion_with_retries(model="gpt-3.5-turbo", messages=[{"content": "What's the capital of China?", "role": "user"}])
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == 'completion_with_retries'
        assert span.kind == SpanKind.INTERNAL
        assert span.attributes[SpanAttributes.LLM_MODEL_NAME] == 'gpt-3.5-turbo'
        assert span.attributes[SpanAttributes.INPUT_VALUE] == "What's the capital of China?"
        assert SpanAttributes.LLM_INVOCATION_PARAMETERS in span.attributes

    # def test_acompletion_with_retries(tracer_provider, in_memory_span_exporter, instrumentor):
    #     import asyncio
    #     asyncio.run(litellm.acompletion_with_retries(model="gpt-3.5-turbo", messages=[{"content": "What's the capital of China?", "role": "user"}]))
    #     spans = in_memory_span_exporter.get_finished_spans()
    #     assert len(spans) == 1
    #     span = spans[0]
    #     assert span.name == 'acompletion_with_retries'
    #     assert span.kind == SpanKind.INTERNAL
    #     assert span.attributes[SpanAttributes.LLM_MODEL_NAME] == 'gpt-3.5-turbo'
    #     assert span.attributes[SpanAttributes.INPUT_VALUE] == "What's the capital of China?"
    #     assert SpanAttributes.LLM_INVOCATION_PARAMETERS in span.attributes

    def test_embedding(tracer_provider, in_memory_span_exporter, instrumentor):
        in_memory_span_exporter.clear()

        litellm.embedding(model='text-embedding-ada-002', input=["good morning from litellm"])
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == 'embedding'
        assert span.kind == SpanKind.INTERNAL
        assert span.attributes[SpanAttributes.EMBEDDING_MODEL_NAME] == 'text-embedding-ada-002'
        assert span.attributes[EmbeddingAttributes.EMBEDDING_TEXT][0] == ["good morning from litellm"][0]
        assert span.attributes[SpanAttributes.INPUT_VALUE] == str(["good morning from litellm"])

    def test_aembedding(tracer_provider, in_memory_span_exporter, instrumentor):
        in_memory_span_exporter.clear()
        import asyncio
        asyncio.run(litellm.aembedding(model='text-embedding-ada-002', input=["good morning from litellm"]))
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == 'aembedding'
        assert span.kind == SpanKind.INTERNAL
        assert span.attributes[SpanAttributes.EMBEDDING_MODEL_NAME] == 'text-embedding-ada-002'
        assert span.attributes[EmbeddingAttributes.EMBEDDING_TEXT][0] == ["good morning from litellm"][0]
        assert span.attributes[SpanAttributes.INPUT_VALUE] == str(["good morning from litellm"])
