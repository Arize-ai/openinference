import json
from unittest.mock import patch

import litellm
import pytest
from litellm.llms.openai import OpenAIChatCompletion
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.semconv.trace import EmbeddingAttributes, ImageAttributes, SpanAttributes
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
    mock_response_embedding = litellm.EmbeddingResponse(
        model="text-embedding-ada-002",
        data=[{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
        object="list",
        usage={"completion_tokens": 1, "prompt_tokens": 6, "total_tokens": 6},
    )

    mock_response_image_gen = litellm.ImageResponse(
        created=1722359754,
        data=[{"b64_json": None, "revised_prompt": None, "url": "https://dummy-url"}],
    )

    def _assert_common_span_attributes_for_completion(
        self, span, model_name, input_value, output_value, prompt_tokens=10, completion_tokens=20
    ):
        assert span.attributes[SpanAttributes.LLM_MODEL_NAME] == model_name
        assert span.attributes[SpanAttributes.INPUT_VALUE] == input_value
        assert span.attributes[SpanAttributes.OUTPUT_VALUE] == output_value
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == prompt_tokens
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == completion_tokens
        assert (
            span.attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL]
            == prompt_tokens + completion_tokens
        )

    def _assert_embedding_attributes(self, span):
        assert span.attributes[SpanAttributes.EMBEDDING_MODEL_NAME] == "text-embedding-ada-002"
        assert (
            span.attributes[EmbeddingAttributes.EMBEDDING_TEXT][0]
            == ["good morning from litellm"][0]
        )
        assert span.attributes[SpanAttributes.INPUT_VALUE] == str(["good morning from litellm"])

        assert span.attributes[EmbeddingAttributes.EMBEDDING_VECTOR] == str([0.1, 0.2, 0.3])
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 6
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 1
        assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 6

    def _assert_image_gen_attributes(self, span):
        assert span.attributes[SpanAttributes.LLM_MODEL_NAME] == "dall-e-2"
        assert span.attributes[SpanAttributes.INPUT_VALUE] == "a sunrise over the mountains"

        assert span.attributes[ImageAttributes.IMAGE_URL] == "https://dummy-url"
        assert span.attributes[SpanAttributes.OUTPUT_VALUE] == "https://dummy-url"

    def test_completion(self, tracer_provider, in_memory_span_exporter, instrumentor):
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
        self._assert_common_span_attributes_for_completion(
            span, "gpt-3.5-turbo", "What's the capital of China?", "Beijing"
        )

    def test_completion_with_parameters(
        self, tracer_provider, in_memory_span_exporter, instrumentor
    ):
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
        self._assert_common_span_attributes_for_completion(
            span, "gpt-3.5-turbo", "What's the capital of China?", "Beijing"
        )
        assert span.attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS] == json.dumps(
            {"mock_response": "Beijing", "temperature": 0.7, "top_p": 0.9}
        )

    def test_completion_with_multiple_messages(
        self, tracer_provider, in_memory_span_exporter, instrumentor
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
        self._assert_common_span_attributes_for_completion(
            span,
            "gpt-3.5-turbo",
            "Hello, I want to bake a cake",
            "Got it! What kind of pie would you like to make?",
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

    async def test_acompletion(self, tracer_provider, in_memory_span_exporter, instrumentor):
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
        self._assert_common_span_attributes_for_completion(
            span, "gpt-3.5-turbo", "What's the capital of China?", "Beijing"
        )

    def test_completion_with_retries(self, tracer_provider, in_memory_span_exporter, instrumentor):
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
        self._assert_common_span_attributes_for_completion(
            span, "gpt-3.5-turbo", "What's the capital of China?", "Beijing"
        )

    # Bug report filed on GitHub for acompletion_with_retries: https://github.com/BerriAI/litellm/issues/4908
    # Until litellm fixes acompletion_with_retries keep this test commented
    # async def test_acompletion_with_retries(self, tracer_provider, in_memory_span_exporter):
    #     in_memory_span_exporter.clear()

    #     await litellm.acompletion_with_retries(
    #         model="gpt-3.5-turbo",
    #         messages=[{"content": "What's the capital of China?", "role": "user"}],
    #         mock_response="Beijing",
    #     )
    #     spans = in_memory_span_exporter.get_finished_spans()
    #     assert len(spans) == 1
    #     span = spans[0]
    #     assert span.name == "acompletion_with_retries"
    #     self._assert_common_span_attributes_for_completion(span,
    #                                                        "gpt-3.5-turbo",
    #                                                        "What's the capital of China?",
    #                                                        "Beijing")

    # Unlike the completion() functions, liteLLM does not offer a mock_response parameter
    # for embeddings or image gen yet
    # For now the following tests monkeypatch OpenAIChatCompletion functions

    def test_embedding(self, tracer_provider, in_memory_span_exporter, instrumentor):
        in_memory_span_exporter.clear()

        with patch.object(
            OpenAIChatCompletion, "embedding", return_value=self.mock_response_embedding
        ):
            litellm.embedding(model="text-embedding-ada-002", input=["good morning from litellm"])

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "embedding"
        self._assert_embedding_attributes(span)

    async def test_aembedding(self, tracer_provider, in_memory_span_exporter, instrumentor):
        in_memory_span_exporter.clear()

        with patch.object(
            OpenAIChatCompletion, "aembedding", return_value=self.mock_response_embedding
        ):
            await litellm.aembedding(
                model="text-embedding-ada-002", input=["good morning from litellm"]
            )

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "aembedding"

        assert span.attributes[SpanAttributes.EMBEDDING_MODEL_NAME] == "text-embedding-ada-002"
        self._assert_embedding_attributes(span)

    def test_image_generation(self, tracer_provider, in_memory_span_exporter, instrumentor):
        in_memory_span_exporter.clear()

        with patch.object(
            OpenAIChatCompletion, "image_generation", return_value=self.mock_response_image_gen
        ):
            litellm.image_generation(
                model="dall-e-2",
                prompt="a sunrise over the mountains",
            )

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "image_generation"
        self._assert_image_gen_attributes(span)

    async def test_aimage_generation(self, tracer_provider, in_memory_span_exporter, instrumentor):
        in_memory_span_exporter.clear()

        with patch.object(
            OpenAIChatCompletion, "aimage_generation", return_value=self.mock_response_image_gen
        ):
            await litellm.aimage_generation(
                model="dall-e-2",
                prompt="a sunrise over the mountains",
            )

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "aimage_generation"
        self._assert_image_gen_attributes(span)

    def test_uninstrument(self, tracer_provider):
        func_names = [
            "completion",
            "acompletion",
            "completion_with_retries",
            # "acompletion_with_retries",
            "embedding",
            "aembedding",
            "image_generation",
            "aimage_generation",
        ]
        # Instrument functions
        instrumentor = LiteLLMInstrumentor(tracer_provider=tracer_provider)
        instrumentor.instrument()
        # Check that the functions are instrumented
        for func_name in func_names:
            instrumented_func = getattr(litellm, func_name)
            assert (
                instrumented_func.__name__ == "_sync_wrapper"
                or instrumented_func.__name__ == "_async_wrapper"
            )
        instrumentor.uninstrument()
        # Test that liteLLM functions are uninstrumented
        for func_name in func_names:
            uninstrumented_func = getattr(litellm, func_name)
            assert uninstrumented_func.__name__ == func_name
        instrumentor.instrument()
        # Check that the functions are re-instrumented
        for func_name in func_names:
            instrumented_func = getattr(litellm, func_name)
            assert (
                instrumented_func.__name__ == "_sync_wrapper"
                or instrumented_func.__name__ == "_async_wrapper"
            )
