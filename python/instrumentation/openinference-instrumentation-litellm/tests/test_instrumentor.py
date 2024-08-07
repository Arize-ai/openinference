import json
from typing import Any, Dict, Generator, List, Mapping, cast
from unittest.mock import patch

import litellm
import pytest
from litellm.llms.openai import OpenAIChatCompletion
from openinference.instrumentation import OITracer, using_attributes
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.semconv.trace import (
    EmbeddingAttributes,
    ImageAttributes,
    SpanAttributes,
)
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue


@pytest.fixture(scope="module")
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(scope="module")
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


# Ensure we're using the common OITracer from common opeinference-instrumentation pkg
def test_oitracer(instrument: Generator[None, None, None]) -> None:
    assert isinstance(LiteLLMInstrumentor()._tracer, OITracer)


@pytest.fixture()
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    LiteLLMInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_completion(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    in_memory_span_exporter.clear()
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"content": "What's the capital of China?", "role": "user"}],
                mock_response="Beijing",
            )
    else:
        litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"content": "What's the capital of China?", "role": "user"}],
            mock_response="Beijing",
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "completion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == "What's the capital of China?"

    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "Beijing"
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    LiteLLMInstrumentor().uninstrument()


def test_completion_with_parameters(
    tracer_provider: TracerProvider, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    in_memory_span_exporter.clear()
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

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
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == "What's the capital of China?"
    assert attributes.get(SpanAttributes.LLM_INVOCATION_PARAMETERS) == json.dumps(
        {"mock_response": "Beijing", "temperature": 0.7, "top_p": 0.9}
    )

    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "Beijing"
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30

    LiteLLMInstrumentor().uninstrument()


def test_completion_with_multiple_messages(
    tracer_provider: TracerProvider, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    in_memory_span_exporter.clear()
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

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
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == "Hello, I want to bake a cake"
    assert attributes.get(SpanAttributes.LLM_INVOCATION_PARAMETERS) == json.dumps(
        {"mock_response": "Got it! What kind of pie would you like to make?"}
    )
    assert attributes.get("input.messages.0.content") == "Hello, I want to bake a cake"
    assert (
        attributes.get("input.messages.1.content") == "Hello, I can pull up some recipes for cakes."
    )
    assert attributes.get("input.messages.2.content") == "No actually I want to make a pie"
    assert attributes.get("input.messages.0.role") == "user"
    assert attributes.get("input.messages.1.role") == "assistant"
    assert attributes.get("input.messages.2.role") == "user"

    assert (
        attributes.get(SpanAttributes.OUTPUT_VALUE)
        == "Got it! What kind of pie would you like to make?"
    )
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30

    LiteLLMInstrumentor().uninstrument()


@pytest.mark.parametrize("use_context_attributes", [False, True])
async def test_acompletion(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    in_memory_span_exporter.clear()
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"content": "What's the capital of China?", "role": "user"}],
                mock_response="Beijing",
            )
    else:
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"content": "What's the capital of China?", "role": "user"}],
            mock_response="Beijing",
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "acompletion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == "What's the capital of China?"

    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "Beijing"
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )

    LiteLLMInstrumentor().uninstrument()


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_completion_with_retries(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    in_memory_span_exporter.clear()
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            litellm.completion_with_retries(
                model="gpt-3.5-turbo",
                messages=[{"content": "What's the capital of China?", "role": "user"}],
                mock_response="Beijing",
            )
    else:
        litellm.completion_with_retries(
            model="gpt-3.5-turbo",
            messages=[{"content": "What's the capital of China?", "role": "user"}],
            mock_response="Beijing",
        )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "completion_with_retries"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == "What's the capital of China?"

    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "Beijing"
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    LiteLLMInstrumentor().uninstrument()


# Bug report filed on GitHub for acompletion_with_retries: https://github.com/BerriAI/litellm/issues/4908
# Until litellm fixes acompletion_with_retries keep this test commented
# async def test_acompletion_with_retries(tracer_provider, in_memory_span_exporter):
#     in_memory_span_exporter.clear()
#
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
# For now the following tests monkeypatch OpenAIChatCompletion functions


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_embedding(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    in_memory_span_exporter.clear()
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

    mock_response_embedding = litellm.EmbeddingResponse(
        model="text-embedding-ada-002",
        data=[{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
        object="list",
        usage={"completion_tokens": 1, "prompt_tokens": 6, "total_tokens": 6},
    )

    with patch.object(OpenAIChatCompletion, "embedding", return_value=mock_response_embedding):
        if use_context_attributes:
            with using_attributes(
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                prompt_template=prompt_template,
                prompt_template_version=prompt_template_version,
                prompt_template_variables=prompt_template_variables,
            ):
                litellm.embedding(
                    model="text-embedding-ada-002", input=["good morning from litellm"]
                )
        else:
            litellm.embedding(model="text-embedding-ada-002", input=["good morning from litellm"])

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embedding"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.EMBEDDING_MODEL_NAME) == "text-embedding-ada-002"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == str(["good morning from litellm"])

    assert attributes.get(EmbeddingAttributes.EMBEDDING_VECTOR) == str([0.1, 0.2, 0.3])
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 6
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 1
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 6

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    LiteLLMInstrumentor().uninstrument()


@pytest.mark.parametrize("use_context_attributes", [False, True])
async def test_aembedding(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    in_memory_span_exporter.clear()
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

    mock_response_embedding = litellm.EmbeddingResponse(
        model="text-embedding-ada-002",
        data=[{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
        object="list",
        usage={"completion_tokens": 1, "prompt_tokens": 6, "total_tokens": 6},
    )

    with patch.object(OpenAIChatCompletion, "aembedding", return_value=mock_response_embedding):
        if use_context_attributes:
            with using_attributes(
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                prompt_template=prompt_template,
                prompt_template_version=prompt_template_version,
                prompt_template_variables=prompt_template_variables,
            ):
                await litellm.aembedding(
                    model="text-embedding-ada-002", input=["good morning from litellm"]
                )
        else:
            await litellm.aembedding(
                model="text-embedding-ada-002", input=["good morning from litellm"]
            )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "aembedding"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.EMBEDDING_MODEL_NAME) == "text-embedding-ada-002"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == str(["good morning from litellm"])

    assert attributes.get(EmbeddingAttributes.EMBEDDING_VECTOR) == str([0.1, 0.2, 0.3])
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 6
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 1
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 6

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )

    LiteLLMInstrumentor().uninstrument()


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_image_generation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    in_memory_span_exporter.clear()
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

    mock_response_image_gen = litellm.ImageResponse(
        created=1722359754,
        data=[{"b64_json": None, "revised_prompt": None, "url": "https://dummy-url"}],
    )

    with patch.object(
        OpenAIChatCompletion, "image_generation", return_value=mock_response_image_gen
    ):
        if use_context_attributes:
            with using_attributes(
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                prompt_template=prompt_template,
                prompt_template_version=prompt_template_version,
                prompt_template_variables=prompt_template_variables,
            ):
                litellm.image_generation(
                    model="dall-e-2",
                    prompt="a sunrise over the mountains",
                )
        else:
            litellm.image_generation(
                model="dall-e-2",
                prompt="a sunrise over the mountains",
            )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "image_generation"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "dall-e-2"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == "a sunrise over the mountains"

    assert attributes.get(ImageAttributes.IMAGE_URL) == "https://dummy-url"
    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "https://dummy-url"

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )

    LiteLLMInstrumentor().uninstrument()


@pytest.mark.parametrize("use_context_attributes", [False, True])
async def test_aimage_generation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    in_memory_span_exporter.clear()
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

    mock_response_image_gen = litellm.ImageResponse(
        created=1722359754,
        data=[{"b64_json": None, "revised_prompt": None, "url": "https://dummy-url"}],
    )
    with patch.object(
        OpenAIChatCompletion, "aimage_generation", return_value=mock_response_image_gen
    ):
        if use_context_attributes:
            with using_attributes(
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                prompt_template=prompt_template,
                prompt_template_version=prompt_template_version,
                prompt_template_variables=prompt_template_variables,
            ):
                await litellm.aimage_generation(
                    model="dall-e-2",
                    prompt="a sunrise over the mountains",
                )
        else:
            await litellm.aimage_generation(
                model="dall-e-2",
                prompt="a sunrise over the mountains",
            )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "aimage_generation"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "dall-e-2"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == "a sunrise over the mountains"

    assert attributes.get(ImageAttributes.IMAGE_URL) == "https://dummy-url"
    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "https://dummy-url"

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )

    LiteLLMInstrumentor().uninstrument()


def test_uninstrument(tracer_provider: TracerProvider) -> None:
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
        assert instrumented_func.is_wrapper

    instrumentor.uninstrument()

    # Test that liteLLM functions are uninstrumented
    for func_name in func_names:
        uninstrumented_func = getattr(litellm, func_name)
        assert uninstrumented_func.__name__ == func_name

    instrumentor.instrument()

    # Check that the functions are re-instrumented
    for func_name in func_names:
        instrumented_func = getattr(litellm, func_name)
        assert instrumented_func.is_wrapper


def _check_context_attributes(
    attributes: Dict[str, Any],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    assert attributes.pop(SpanAttributes.SESSION_ID, None) == session_id
    assert attributes.pop(SpanAttributes.USER_ID, None) == user_id
    attr_metadata = attributes.pop(SpanAttributes.METADATA, None)
    assert attr_metadata is not None
    assert isinstance(attr_metadata, str)  # must be json string
    metadata_dict = json.loads(attr_metadata)
    assert metadata_dict == metadata
    attr_tags = attributes.pop(SpanAttributes.TAG_TAGS, None)
    assert attr_tags is not None
    assert len(attr_tags) == len(tags)
    assert list(attr_tags) == tags
    assert attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE, None) == prompt_template
    assert (
        attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION, None) == prompt_template_version
    )
    assert attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES, None) == json.dumps(
        prompt_template_variables
    )


@pytest.fixture()
def session_id() -> str:
    return "my-test-session-id"


@pytest.fixture()
def user_id() -> str:
    return "my-test-user-id"


@pytest.fixture()
def metadata() -> Dict[str, Any]:
    return {
        "test-int": 1,
        "test-str": "string",
        "test-list": [1, 2, 3],
        "test-dict": {
            "key-1": "val-1",
            "key-2": "val-2",
        },
    }


@pytest.fixture()
def tags() -> List[str]:
    return ["tag-1", "tag-2"]


@pytest.fixture
def prompt_template() -> str:
    return (
        "This is a test prompt template with int {var_int}, "
        "string {var_string}, and list {var_list}"
    )


@pytest.fixture
def prompt_template_version() -> str:
    return "v1.0"


@pytest.fixture
def prompt_template_variables() -> Dict[str, Any]:
    return {
        "var_int": 1,
        "var_str": "2",
        "var_list": [1, 2, 3],
    }
