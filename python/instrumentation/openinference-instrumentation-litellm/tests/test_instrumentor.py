import json

import litellm
import pytest
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.semconv.trace import (
    EmbeddingAttributes,
    SpanAttributes,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


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

        litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"content": "What's the capital of China?", "role": "user"}],
            mock_response="Beijing",
        )

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "completion"
        assert span.attributes[SpanAttributes.LLM_MODEL_NAME] == "gpt-3.5-turbo"
        assert span.attributes[SpanAttributes.INPUT_VALUE] == "What's the capital of China?"

        assert span.attributes[SpanAttributes.OUTPUT_VALUE] == "Beijing"
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 10
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 20
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 30

    def test_completion_with_parameters(tracer_provider, in_memory_span_exporter, instrumentor):
        in_memory_span_exporter.clear()

        litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"content": "What's the capital of China?", "role": "user"}],
            mock_response="Beijing",
            temperature=0.7,
            top_p=0.9,
        )
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "completion"
        assert span.attributes[SpanAttributes.LLM_MODEL_NAME] == "gpt-3.5-turbo"
        assert span.attributes[SpanAttributes.INPUT_VALUE] == "What's the capital of China?"
        assert span.attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS] == json.dumps(
            {"mock_response": "Beijing", "temperature": 0.7, "top_p": 0.9}
        )

        assert span.attributes[SpanAttributes.OUTPUT_VALUE] == "Beijing"
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 10
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 20
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 30

    def test_completion_with_multiple_messages(
        tracer_provider, in_memory_span_exporter, instrumentor
    ):
        in_memory_span_exporter.clear()

        litellm.completion(
            model="gpt-3.5-turbo",
            messages=[
                {"content": "Hello, I want to bake a cake", "role": "user"},
                {"content": "Hello, I can pull up some recipes for cakes.", "role": "assistant"},
                {"content": "No actually I want to make a pie", "role": "user"},
            ],
            mock_response="Got it! What kind of pie would you like to make?",
        )
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "completion"
        assert span.attributes[SpanAttributes.LLM_MODEL_NAME] == "gpt-3.5-turbo"
        assert span.attributes[SpanAttributes.INPUT_VALUE] == "Hello, I want to bake a cake"
        assert span.attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS] == json.dumps(
            {"mock_response": "Got it! What kind of pie would you like to make?"}
        )
        assert span.attributes["input.messages.0.content"] == "Hello, I want to bake a cake"
        assert (
            span.attributes["input.messages.1.content"]
            == "Hello, I can pull up some recipes for cakes."
        )
        assert span.attributes["input.messages.2.content"] == "No actually I want to make a pie"
        assert span.attributes["input.messages.0.role"] == "user"
        assert span.attributes["input.messages.1.role"] == "assistant"
        assert span.attributes["input.messages.2.role"] == "user"

        assert (
            span.attributes[SpanAttributes.OUTPUT_VALUE]
            == "Got it! What kind of pie would you like to make?"
        )
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 10
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 20
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 30

    async def test_acompletion(tracer_provider, in_memory_span_exporter, instrumentor):
        in_memory_span_exporter.clear()

        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"content": "What's the capital of China?", "role": "user"}],
            mock_response="Beijing",
        )

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "acompletion"
        assert span.attributes[SpanAttributes.LLM_MODEL_NAME] == "gpt-3.5-turbo"
        assert span.attributes[SpanAttributes.INPUT_VALUE] == "What's the capital of China?"

        assert span.attributes[SpanAttributes.OUTPUT_VALUE] == "Beijing"
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 10
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 20
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 30

    def test_completion_with_retries(tracer_provider, in_memory_span_exporter, instrumentor):
        in_memory_span_exporter.clear()

        litellm.completion_with_retries(
            model="gpt-3.5-turbo",
            messages=[{"content": "What's the capital of China?", "role": "user"}],
            mock_response="Beijing",
        )
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "completion_with_retries"
        assert span.attributes[SpanAttributes.LLM_MODEL_NAME] == "gpt-3.5-turbo"
        assert span.attributes[SpanAttributes.INPUT_VALUE] == "What's the capital of China?"

        assert span.attributes[SpanAttributes.OUTPUT_VALUE] == "Beijing"
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 10
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 20
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 30

    # Bug report filed on GitHub for acompletion_with_retries: https://github.com/BerriAI/litellm/issues/4908
    # Until litellm fixes acompletion_with_retries keep this test commented
    # async def test_acompletion_with_retries(tracer_provider, in_memory_span_exporter):
    #     await litellm.acompletion_with_retries(
    #         model="gpt-3.5-turbo",
    #         messages=[{"content": "What's the capital of China?", "role": "user"}],
    #     )
    #     spans = in_memory_span_exporter.get_finished_spans()
    #     assert len(spans) == 1
    #     span = spans[0]
    #     assert span.name == "acompletion_with_retries"
    #     assert span.attributes[SpanAttributes.LLM_MODEL_NAME] == "gpt-3.5-turbo"
    #     assert span.attributes[SpanAttributes.INPUT_VALUE] == "What's the capital of China?"

    #     assert span.attributes[SpanAttributes.OUTPUT_VALUE] == "Beijing"
    #     assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 10
    #     assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 20
    #     assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 30

    # Unlike the completion() functions, liteLLM does not offer a mock_response parameter
    # for embeddings or image gen yet
    # For now the following test functions make real calls that require Internet connection

    def test_embedding(tracer_provider, in_memory_span_exporter, instrumentor):
        in_memory_span_exporter.clear()

        litellm.embedding(model="text-embedding-ada-002", input=["good morning from litellm"])
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "embedding"
        assert span.attributes[SpanAttributes.EMBEDDING_MODEL_NAME] == "text-embedding-ada-002"
        assert (
            span.attributes[EmbeddingAttributes.EMBEDDING_TEXT][0]
            == ["good morning from litellm"][0]
        )
        assert span.attributes[SpanAttributes.INPUT_VALUE] == str(["good morning from litellm"])

    async def test_aembedding(tracer_provider, in_memory_span_exporter, instrumentor):
        in_memory_span_exporter.clear()
        await litellm.aembedding(
            model="text-embedding-ada-002", input=["good morning from litellm"]
        )

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "aembedding"
        assert span.attributes[SpanAttributes.EMBEDDING_MODEL_NAME] == "text-embedding-ada-002"
        assert (
            span.attributes[EmbeddingAttributes.EMBEDDING_TEXT][0]
            == ["good morning from litellm"][0]
        )
        assert span.attributes[SpanAttributes.INPUT_VALUE] == str(["good morning from litellm"])

    def test_image_generation(tracer_provider, in_memory_span_exporter, instrumentor):
        in_memory_span_exporter.clear()

        litellm.image_generation(
            model="dall-e-2",
            prompt="a sunrise over the mountains",
            mock_response="mock response url",
        )
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "image_generation"
        assert span.attributes[SpanAttributes.LLM_MODEL_NAME] == "dall-e-2"
        assert span.attributes[SpanAttributes.INPUT_VALUE] == "a sunrise over the mountains"

    async def test_aimage_generation(tracer_provider, in_memory_span_exporter, instrumentor):
        in_memory_span_exporter.clear()
        await litellm.aimage_generation(
            model="dall-e-2",
            prompt="a sunrise over the mountains",
            mock_response="mock response url",
        )

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "aimage_generation"
        assert span.attributes[SpanAttributes.LLM_MODEL_NAME] == "dall-e-2"
        assert span.attributes[SpanAttributes.INPUT_VALUE] == "a sunrise over the mountains"

    
