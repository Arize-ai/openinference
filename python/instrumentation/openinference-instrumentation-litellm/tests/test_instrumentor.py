import json
from typing import Any, Dict, Generator, List, Mapping, Optional, Union, cast
from unittest.mock import patch

import litellm
import pytest
from litellm import OpenAIChatCompletion  # type: ignore[attr-defined]
from litellm.types.utils import EmbeddingResponse, ImageObject, ImageResponse, Usage
from litellm.types.utils import Message as LitellmMessage
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import OITracer, safe_json_dumps, using_attributes
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.semconv.trace import (
    EmbeddingAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)


@pytest.fixture(scope="module")
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(scope="module")
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def setup_litellm_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    LiteLLMInstrumentor().uninstrument()


class TestInstrumentor:
    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        (instrumentor_entrypoint,) = entry_points(
            group="opentelemetry_instrumentor", name="litellm"
        )
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, LiteLLMInstrumentor)

    # Ensure we're using the common OITracer from common openinference-instrumentation pkg
    def test_oitracer(self, setup_litellm_instrumentation: Any) -> None:
        assert isinstance(LiteLLMInstrumentor()._tracer, OITracer)


@pytest.mark.parametrize("use_context_attributes", [False, True])
@pytest.mark.parametrize("n", [1, 5])
@pytest.mark.parametrize(
    "input_messages",
    [
        [{"content": "What's the capital of China?", "role": "user"}],
        [LitellmMessage(content="How can I help you?", role="assistant")],
    ],
)
def test_completion(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    use_context_attributes: bool,
    input_messages: List[Union[Dict[str, Any], LitellmMessage]],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
    n: int,
) -> None:
    in_memory_span_exporter.clear()

    response = None
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
            response = litellm.completion(
                model="gpt-3.5-turbo",
                messages=input_messages,
                n=n,
                mock_response="Beijing",
            )
    else:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=input_messages,
            n=n,
            mock_response="Beijing",
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "completion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    input_values = [
        msg.json() if isinstance(msg, LitellmMessage) else msg  # type: ignore[no-untyped-call]
        for msg in input_messages
    ]
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps({"messages": input_values})
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "Beijing"
    for i, choice in enumerate(response["choices"]):
        _check_llm_message(SpanAttributes.LLM_OUTPUT_MESSAGES, i, attributes, choice.message)

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


@pytest.mark.parametrize("use_context_attributes", [True])
@pytest.mark.parametrize("n", [1])
def test_completion_sync_streaming(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
    n: int,
) -> None:
    in_memory_span_exporter.clear()

    input_messages = [{"content": "What's the capital of China?", "role": "user"}]
    response = None
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
            response = litellm.completion(
                model="gpt-3.5-turbo",
                messages=input_messages,
                mock_response="The capital of China is Beijing",
                n=n,
                stream=True,
            )
    else:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=input_messages,
            mock_response="The capital of China is Beijing",
            n=n,
            stream=True,
        )

    output_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            output_message += chunk.choices[0].delta.content

    assert output_message == "The capital of China is Beijing"

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "completion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"

    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "The capital of China is Beijing"

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


def test_completion_with_parameters(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    input_messages = [{"content": "What's the capital of China?", "role": "user"}]
    litellm.completion(
        model="gpt-3.5-turbo",
        messages=input_messages,
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
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert attributes.get(SpanAttributes.LLM_INVOCATION_PARAMETERS) == json.dumps(
        {"mock_response": "Beijing", "temperature": 0.7, "top_p": 0.9}
    )

    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "Beijing"
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30


def test_completion_with_multiple_messages(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    input_messages = [
        {"content": "Hello, I want to bake a cake", "role": "user"},
        {"content": "Hello, I can pull up some recipes for cakes.", "role": "assistant"},
        {"content": "No actually I want to make a pie", "role": "user"},
    ]
    litellm.completion(
        model="gpt-3.5-turbo",
        messages=input_messages,
        mock_response="Got it! What kind of pie would you like to make?",
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "completion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    for i, message in enumerate(input_messages):
        _check_llm_message(SpanAttributes.LLM_INPUT_MESSAGES, i, attributes, message)
    assert attributes.get(SpanAttributes.LLM_INVOCATION_PARAMETERS) == json.dumps(
        {"mock_response": "Got it! What kind of pie would you like to make?"}
    )
    assert (
        attributes.get(SpanAttributes.OUTPUT_VALUE)
        == "Got it! What kind of pie would you like to make?"
    )
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30


def test_completion_image_support(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    input_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://dummy_image.jpg"},
                },
            ],
        }
    ]
    litellm.completion(
        model="gpt-4o",
        messages=input_messages,
        mock_response="That's an image of a pasture",
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "completion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    for i, message in enumerate(input_messages):
        _check_llm_message(SpanAttributes.LLM_INPUT_MESSAGES, i, attributes, message)
    assert attributes.get(SpanAttributes.LLM_INVOCATION_PARAMETERS) == json.dumps(
        {"mock_response": "That's an image of a pasture"}
    )
    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "That's an image of a pasture"
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30


@pytest.mark.parametrize("use_context_attributes", [False, True])
async def test_acompletion(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
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

    input_messages = [{"content": "What's the capital of China?", "role": "user"}]
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
                messages=input_messages,
                mock_response="Beijing",
            )
    else:
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=input_messages,
            mock_response="Beijing",
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "acompletion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"

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


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_completion_with_retries(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
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

    input_messages = [{"content": "What's the capital of China?", "role": "user"}]
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
            litellm.completion_with_retries(  # type: ignore [no-untyped-call]
                model="gpt-3.5-turbo",
                messages=input_messages,
                mock_response="Beijing",
            )
    else:
        litellm.completion_with_retries(  # type: ignore [no-untyped-call]
            model="gpt-3.5-turbo",
            messages=input_messages,
            mock_response="Beijing",
        )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "completion_with_retries"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"

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
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
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

    mock_response_embedding = EmbeddingResponse(
        model="text-embedding-ada-002",
        data=[{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
        object="list",
        usage=Usage(prompt_tokens=6, completion_tokens=1, total_tokens=6),
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


@pytest.mark.parametrize("use_context_attributes", [False, True])
async def test_aembedding(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
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

    mock_response_embedding = EmbeddingResponse(
        model="text-embedding-ada-002",
        data=[{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
        object="list",
        usage=Usage(prompt_tokens=6, completion_tokens=1, total_tokens=6),
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


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_image_generation_url(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
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

    mock_response_image_gen = ImageResponse(
        created=1722359754,
        data=[ImageObject(b64_json=None, revised_prompt=None, url="https://dummy-url")],  # type: ignore
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


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_image_generation_b64json(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
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

    mock_response_image_gen = ImageResponse(
        created=1722359754,
        data=[ImageObject(b64_json="dummy_b64_json", revised_prompt=None, url=None)],  # type: ignore
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

    assert attributes.get(ImageAttributes.IMAGE_URL) == "dummy_b64_json"
    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "dummy_b64_json"

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


@pytest.mark.parametrize("use_context_attributes", [False, True])
async def test_aimage_generation(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
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

    mock_response_image_gen = ImageResponse(
        created=1722359754,
        data=[ImageObject(b64_json=None, revised_prompt=None, url="https://dummy-url")],  # type: ignore
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


def _check_llm_message(
    prefix: str,
    i: int,
    attributes: Dict[str, Any],
    message: Dict[str, Any],
    hide_text: bool = False,
    hide_images: bool = False,
    image_limit: Optional[int] = None,
) -> None:
    assert attributes.pop(message_role(prefix, i), None) == message.get("role")
    expected_content = message.get("content")
    if isinstance(expected_content, list):
        for j, expected_content_item in enumerate(expected_content):
            content_item_type = attributes.pop(message_contents_type(prefix, i, j), None)
            expected_content_item_type = expected_content_item.get("type")
            if expected_content_item_type == "image_url":
                expected_content_item_type = "image"
            assert content_item_type == expected_content_item_type
            if content_item_type == "text":
                content_item_text = attributes.pop(message_contents_text(prefix, i, j), None)
                if hide_text:
                    assert content_item_text == REDACTED_VALUE
                else:
                    assert content_item_text == expected_content_item.get("text")
            elif content_item_type == "image":
                content_item_image_url = attributes.pop(
                    message_contents_image_url(prefix, i, j), None
                )
                if hide_images:
                    assert content_item_image_url is None
                else:
                    expected_url = expected_content_item.get("image_url").get("url")
                    if image_limit is not None and len(expected_url) > image_limit:
                        assert content_item_image_url == REDACTED_VALUE
                    else:
                        assert content_item_image_url == expected_url
    else:
        content = attributes.pop(message_content(prefix, i), None)
        if expected_content is not None and hide_text:
            assert content == REDACTED_VALUE
        else:
            assert content == expected_content


def message_content(prefix: str, i: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENT}"


def message_role(prefix: str, i: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_ROLE}"


def message_contents_type(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENTS}.{j}.{MESSAGE_CONTENT_TYPE}"


def message_contents_text(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENTS}.{j}.{MESSAGE_CONTENT_TEXT}"


def message_contents_image_url(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENTS}.{j}.{MESSAGE_CONTENT_IMAGE}.{IMAGE_URL}"


MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
IMAGE_URL = ImageAttributes.IMAGE_URL
REDACTED_VALUE = "__REDACTED__"
