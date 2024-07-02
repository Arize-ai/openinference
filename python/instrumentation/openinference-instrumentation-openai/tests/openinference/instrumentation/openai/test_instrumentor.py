import asyncio
import json
import logging
import random
from contextlib import suppress
from importlib import import_module
from importlib.metadata import version
from itertools import count
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Mapping,
    Sequence,
    Tuple,
    Union,
    cast,
)

import pytest
from httpx import AsyncByteStream, Response
from openinference.instrumentation import using_attributes
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import (
    EmbeddingAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue
from respx import MockRouter

for name, logger in logging.root.manager.loggerDict.items():
    if name.startswith("openinference.") and isinstance(logger, logging.Logger):
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler())


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("is_raw", [False, True])
@pytest.mark.parametrize("is_stream", [False, True])
@pytest.mark.parametrize("status_code", [200, 400])
@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_chat_completions(
    is_async: bool,
    is_raw: bool,
    is_stream: bool,
    status_code: int,
    use_context_attributes: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    completion_usage: Dict[str, Any],
    model_name: str,
    chat_completion_mock_stream: Tuple[List[bytes], List[Dict[str, Any]]],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    input_messages: List[Dict[str, Any]] = get_messages()
    output_messages: List[Dict[str, Any]] = (
        chat_completion_mock_stream[1] if is_stream else get_messages()
    )
    invocation_parameters = {
        "stream": is_stream,
        "model": randstr(),
        "temperature": random.random(),
        "n": len(output_messages),
    }
    url = "https://api.openai.com/v1/chat/completions"
    respx_kwargs: Dict[str, Any] = {
        **(
            {"stream": MockAsyncByteStream(chat_completion_mock_stream[0])}
            if is_stream
            else {
                "json": {
                    "choices": [
                        {"index": i, "message": message, "finish_reason": "stop"}
                        for i, message in enumerate(output_messages)
                    ],
                    "model": model_name,
                    "usage": completion_usage,
                }
            }
        ),
    }
    respx_mock.post(url).mock(return_value=Response(status_code=status_code, **respx_kwargs))
    create_kwargs = {"messages": input_messages, **invocation_parameters}
    openai = import_module("openai")
    completions = (
        openai.AsyncOpenAI(api_key="sk-").chat.completions
        if is_async
        else openai.OpenAI(api_key="sk-").chat.completions
    )
    create = completions.with_raw_response.create if is_raw else completions.create

    async def task() -> None:
        response = await create(**create_kwargs)
        response = response.parse() if is_raw else response
        if is_stream:
            async for _ in response:
                pass

    with suppress(openai.BadRequestError):
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
                if is_async:
                    asyncio.run(task())
                else:
                    response = create(**create_kwargs)
                    response = response.parse() if is_raw else response
                    if is_stream:
                        for _ in response:
                            pass
        else:
            if is_async:
                asyncio.run(task())
            else:
                response = create(**create_kwargs)
                response = response.parse() if is_raw else response
                if is_stream:
                    for _ in response:
                        pass
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2  # first span should be from the httpx instrumentor
    span: ReadableSpan = spans[1]
    if status_code == 200:
        assert span.status.is_ok
        assert not span.status.description
    elif status_code == 400:
        assert not span.status.is_ok and not span.status.is_unset
        assert span.status.description and span.status.description.startswith(
            openai.BadRequestError.__name__
        )
        assert len(span.events) == 1
        event = span.events[0]
        assert event.name == "exception"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(OPENINFERENCE_SPAN_KIND, None) == OpenInferenceSpanKindValues.LLM.value
    assert isinstance(attributes.pop(INPUT_VALUE, None), str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE, None))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert (
        json.loads(cast(str, attributes.pop(LLM_INVOCATION_PARAMETERS, None)))
        == invocation_parameters
    )
    for prefix, messages in (
        (LLM_INPUT_MESSAGES, input_messages),
        *(((LLM_OUTPUT_MESSAGES, output_messages),) if status_code == 200 else ()),
    ):
        for i, message in enumerate(messages):
            assert attributes.pop(message_role(prefix, i), None) == message.get("role")
            assert attributes.pop(message_content(prefix, i), None) == message.get("content")
            if function_call := message.get("function_call"):
                assert attributes.pop(
                    message_function_call_name(prefix, i), None
                ) == function_call.get("name")
                assert attributes.pop(
                    message_function_call_arguments(prefix, i), None
                ) == function_call.get("arguments")
            if _openai_version() >= (1, 1, 0) and (tool_calls := message.get("tool_calls")):
                for j, tool_call in enumerate(tool_calls):
                    if function := tool_call.get("function"):
                        assert attributes.pop(
                            tool_call_function_name(prefix, i, j), None
                        ) == function.get("name")
                        assert attributes.pop(
                            tool_call_function_arguments(prefix, i, j), None
                        ) == function.get("arguments")
    if status_code == 200:
        assert isinstance(attributes.pop(OUTPUT_VALUE, None), str)
        assert (
            OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE, None))
            == OpenInferenceMimeTypeValues.JSON
        )
        if not is_stream:
            # Usage is not available for streaming in general.
            assert attributes.pop(LLM_TOKEN_COUNT_TOTAL, None) == completion_usage["total_tokens"]
            assert attributes.pop(LLM_TOKEN_COUNT_PROMPT, None) == completion_usage["prompt_tokens"]
            assert (
                attributes.pop(LLM_TOKEN_COUNT_COMPLETION, None)
                == completion_usage["completion_tokens"]
            )
            # We left out model_name from our mock stream.
            assert attributes.pop(LLM_MODEL_NAME, None) == model_name
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
    assert attributes == {}  # test should account for all span attributes


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("is_raw", [False, True])
@pytest.mark.parametrize("is_stream", [False, True])
@pytest.mark.parametrize("status_code", [200, 400])
@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_completions(
    is_async: bool,
    is_raw: bool,
    is_stream: bool,
    status_code: int,
    use_context_attributes: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    completion_usage: Dict[str, Any],
    model_name: str,
    completion_mock_stream: Tuple[List[bytes], List[str]],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    prompt: List[str] = get_texts()
    output_texts: List[str] = completion_mock_stream[1] if is_stream else get_texts()
    invocation_parameters = {
        "stream": is_stream,
        "model": randstr(),
        "temperature": random.random(),
        "n": len(output_texts),
    }
    url = "https://api.openai.com/v1/completions"
    respx_kwargs: Dict[str, Any] = {
        **(
            {"stream": MockAsyncByteStream(completion_mock_stream[0])}
            if is_stream
            else {
                "json": {
                    "choices": [
                        {"index": i, "text": text, "finish_reason": "stop"}
                        for i, text in enumerate(output_texts)
                    ],
                    "model": model_name,
                    "usage": completion_usage,
                }
            }
        ),
    }
    respx_mock.post(url).mock(return_value=Response(status_code=status_code, **respx_kwargs))
    create_kwargs = {"prompt": prompt, **invocation_parameters}
    openai = import_module("openai")
    completions = (
        openai.AsyncOpenAI(api_key="sk-").completions
        if is_async
        else openai.OpenAI(api_key="sk-").completions
    )
    create = completions.with_raw_response.create if is_raw else completions.create

    async def task() -> None:
        response = await create(**create_kwargs)
        response = response.parse() if is_raw else response
        if is_stream:
            async for _ in response:
                pass

    with suppress(openai.BadRequestError):
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
                if is_async:
                    asyncio.run(task())
                else:
                    response = create(**create_kwargs)
                    response = response.parse() if is_raw else response
                    if is_stream:
                        for _ in response:
                            pass
        else:
            if is_async:
                asyncio.run(task())
            else:
                response = create(**create_kwargs)
                response = response.parse() if is_raw else response
                if is_stream:
                    for _ in response:
                        pass
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2  # first span should be from the httpx instrumentor
    span: ReadableSpan = spans[1]
    if status_code == 200:
        assert span.status.is_ok
        assert not span.status.description
    elif status_code == 400:
        assert not span.status.is_ok and not span.status.is_unset
        assert span.status.description and span.status.description.startswith(
            openai.BadRequestError.__name__
        )
        assert len(span.events) == 1
        event = span.events[0]
        assert event.name == "exception"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(OPENINFERENCE_SPAN_KIND, None) == OpenInferenceSpanKindValues.LLM.value
    assert (
        json.loads(cast(str, attributes.pop(LLM_INVOCATION_PARAMETERS, None)))
        == invocation_parameters
    )
    assert isinstance(attributes.pop(INPUT_VALUE, None), str)
    assert isinstance(attributes.pop(INPUT_MIME_TYPE, None), str)
    if status_code == 200:
        assert isinstance(attributes.pop(OUTPUT_VALUE, None), str)
        assert isinstance(attributes.pop(OUTPUT_MIME_TYPE, None), str)
        assert list(cast(Sequence[str], attributes.pop(LLM_PROMPTS, None))) == prompt
        if not is_stream:
            # Usage is not available for streaming in general.
            assert attributes.pop(LLM_TOKEN_COUNT_TOTAL, None) == completion_usage["total_tokens"]
            assert attributes.pop(LLM_TOKEN_COUNT_PROMPT, None) == completion_usage["prompt_tokens"]
            assert (
                attributes.pop(LLM_TOKEN_COUNT_COMPLETION, None)
                == completion_usage["completion_tokens"]
            )
            # We left out model_name from our mock stream.
            assert attributes.pop(LLM_MODEL_NAME, None) == model_name
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
    assert attributes == {}  # test should account for all span attributes


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("is_raw", [False, True])
@pytest.mark.parametrize("status_code", [200, 400])
@pytest.mark.parametrize("encoding_format", ["float", "base64"])
@pytest.mark.parametrize("input_text", ["hello", ["hello", "world"]])
def test_embeddings(
    is_async: bool,
    is_raw: bool,
    encoding_format: str,
    input_text: Union[str, List[str]],
    status_code: int,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    model_name: str,
) -> None:
    invocation_parameters = {
        "model": randstr(),
        "encoding_format": encoding_format,
    }
    embedding_model_name = randstr()
    embedding_usage = {
        "prompt_tokens": random.randint(10, 100),
        "total_tokens": random.randint(10, 100),
    }
    output_embeddings = [("AACAPwAAAEA=", (1.0, 2.0)), ((2.0, 3.0), (2.0, 3.0))]
    url = "https://api.openai.com/v1/embeddings"
    respx_mock.post(url).mock(
        return_value=Response(
            status_code=status_code,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": i, "embedding": embedding[0]}
                    for i, embedding in enumerate(output_embeddings)
                ],
                "model": embedding_model_name,
                "usage": embedding_usage,
            },
        )
    )
    create_kwargs = {"input": input_text, **invocation_parameters}
    openai = import_module("openai")
    completions = (
        openai.AsyncOpenAI(api_key="sk-").embeddings
        if is_async
        else openai.OpenAI(api_key="sk-").embeddings
    )
    create = completions.with_raw_response.create if is_raw else completions.create
    with suppress(openai.BadRequestError):
        if is_async:

            async def task() -> None:
                response = await create(**create_kwargs)
                _ = response.parse() if is_raw else response

            asyncio.run(task())
        else:
            response = create(**create_kwargs)
            _ = response.parse() if is_raw else response
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2  # first span should be from the httpx instrumentor
    span: ReadableSpan = spans[1]
    if status_code == 200:
        assert span.status.is_ok
        assert not span.status.description
    elif status_code == 400:
        assert not span.status.is_ok and not span.status.is_unset
        assert span.status.description and span.status.description.startswith(
            openai.BadRequestError.__name__
        )
        assert len(span.events) == 1
        event = span.events[0]
        assert event.name == "exception"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert (
        attributes.pop(OPENINFERENCE_SPAN_KIND, None) == OpenInferenceSpanKindValues.EMBEDDING.value
    )
    assert (
        json.loads(cast(str, attributes.pop(LLM_INVOCATION_PARAMETERS, None)))
        == invocation_parameters
    )
    assert isinstance(attributes.pop(INPUT_VALUE, None), str)
    assert isinstance(attributes.pop(INPUT_MIME_TYPE, None), str)
    if status_code == 200:
        assert isinstance(attributes.pop(OUTPUT_VALUE, None), str)
        assert isinstance(attributes.pop(OUTPUT_MIME_TYPE, None), str)
        assert attributes.pop(EMBEDDING_MODEL_NAME, None) == embedding_model_name
        assert attributes.pop(LLM_TOKEN_COUNT_TOTAL, None) == embedding_usage["total_tokens"]
        assert attributes.pop(LLM_TOKEN_COUNT_PROMPT, None) == embedding_usage["prompt_tokens"]
        for i, text in enumerate(input_text if isinstance(input_text, list) else [input_text]):
            assert attributes.pop(f"{EMBEDDING_EMBEDDINGS}.{i}.{EMBEDDING_TEXT}", None) == text
        for i, embedding in enumerate(output_embeddings):
            assert (
                attributes.pop(f"{EMBEDDING_EMBEDDINGS}.{i}.{EMBEDDING_VECTOR}", None)
                == embedding[1]
            )
    assert attributes == {}  # test should account for all span attributes


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("is_raw", [False, True])
@pytest.mark.parametrize("is_stream", [False, True])
@pytest.mark.parametrize("status_code", [200, 400])
@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_chat_completions_with_multiple_message_contents(
    is_async: bool,
    is_raw: bool,
    is_stream: bool,
    status_code: int,
    use_context_attributes: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    completion_usage: Dict[str, Any],
    model_name: str,
    chat_completion_mock_stream: Tuple[List[bytes], List[Dict[str, Any]]],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    input_messages: List[Dict[str, Any]] = get_messages_with_multiple_contents()
    output_messages: List[Dict[str, Any]] = (
        chat_completion_mock_stream[1] if is_stream else get_messages()
    )
    invocation_parameters = {
        "stream": is_stream,
        "model": model_name,
        "temperature": random.random(),
        "n": len(output_messages),
    }
    url = "https://api.openai.com/v1/chat/completions"
    respx_kwargs: Dict[str, Any] = {
        **(
            {"stream": MockAsyncByteStream(chat_completion_mock_stream[0])}
            if is_stream
            else {
                "json": {
                    "choices": [
                        {"index": i, "message": message, "finish_reason": "stop"}
                        for i, message in enumerate(output_messages)
                    ],
                    "model": model_name,
                    "usage": completion_usage,
                }
            }
        ),
    }
    respx_mock.post(url).mock(return_value=Response(status_code=status_code, **respx_kwargs))
    create_kwargs = {"messages": input_messages, **invocation_parameters}
    openai = import_module("openai")
    completions = (
        openai.AsyncOpenAI(api_key="sk-").chat.completions
        if is_async
        else openai.OpenAI(api_key="sk-").chat.completions
    )
    create = completions.with_raw_response.create if is_raw else completions.create

    async def task() -> None:
        response = await create(**create_kwargs)
        response = response.parse() if is_raw else response
        if is_stream:
            async for _ in response:
                pass

    with suppress(openai.BadRequestError):
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
                if is_async:
                    asyncio.run(task())
                else:
                    response = create(**create_kwargs)
                    response = response.parse() if is_raw else response
                    if is_stream:
                        for _ in response:
                            pass
        else:
            if is_async:
                asyncio.run(task())
            else:
                response = create(**create_kwargs)
                response = response.parse() if is_raw else response
                if is_stream:
                    for _ in response:
                        pass
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2  # first span should be from the httpx instrumentor
    span: ReadableSpan = spans[1]
    if status_code == 200:
        assert span.status.is_ok
        assert not span.status.description
    elif status_code == 400:
        assert not span.status.is_ok and not span.status.is_unset
        assert span.status.description and span.status.description.startswith(
            openai.BadRequestError.__name__
        )
        assert len(span.events) == 1
        event = span.events[0]
        assert event.name == "exception"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(OPENINFERENCE_SPAN_KIND, None) == OpenInferenceSpanKindValues.LLM.value
    assert isinstance(attributes.pop(INPUT_VALUE, None), str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE, None))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert (
        json.loads(cast(str, attributes.pop(LLM_INVOCATION_PARAMETERS, None)))
        == invocation_parameters
    )
    for prefix, messages in (
        (LLM_INPUT_MESSAGES, input_messages),
        *(((LLM_OUTPUT_MESSAGES, output_messages),) if status_code == 200 else ()),
    ):
        for i, message in enumerate(messages):
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
                        content_item_text = attributes.pop(
                            message_contents_text(prefix, i, j), None
                        )
                        assert content_item_text == expected_content_item.get("text")
                    elif content_item_type == "image":
                        content_item_image_url = attributes.pop(
                            message_contents_image_url(prefix, i, j), None
                        )
                        assert content_item_image_url == expected_content_item.get("image_url").get(
                            "url"
                        )

            else:
                content = attributes.pop(message_content(prefix, i), None)
                assert content == expected_content

            if function_call := message.get("function_call"):
                assert attributes.pop(
                    message_function_call_name(prefix, i), None
                ) == function_call.get("name")
                assert attributes.pop(
                    message_function_call_arguments(prefix, i), None
                ) == function_call.get("arguments")
            if _openai_version() >= (1, 1, 0) and (tool_calls := message.get("tool_calls")):
                for j, tool_call in enumerate(tool_calls):
                    if function := tool_call.get("function"):
                        assert attributes.pop(
                            tool_call_function_name(prefix, i, j), None
                        ) == function.get("name")
                        assert attributes.pop(
                            tool_call_function_arguments(prefix, i, j), None
                        ) == function.get("arguments")
    if status_code == 200:
        assert isinstance(attributes.pop(OUTPUT_VALUE, None), str)
        assert (
            OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE, None))
            == OpenInferenceMimeTypeValues.JSON
        )
        if not is_stream:
            # Usage is not available for streaming in general.
            assert attributes.pop(LLM_TOKEN_COUNT_TOTAL, None) == completion_usage["total_tokens"]
            assert attributes.pop(LLM_TOKEN_COUNT_PROMPT, None) == completion_usage["prompt_tokens"]
            assert (
                attributes.pop(LLM_TOKEN_COUNT_COMPLETION, None)
                == completion_usage["completion_tokens"]
            )
            # We left out model_name from our mock stream.
            assert attributes.pop(LLM_MODEL_NAME, None) == model_name
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
    assert attributes == {}  # test should account for all span attributes


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
    assert attributes.pop(SESSION_ID, None) == session_id
    assert attributes.pop(USER_ID, None) == user_id
    attr_metadata = attributes.pop(METADATA, None)
    assert attr_metadata is not None
    assert isinstance(attr_metadata, str)  # must be json string
    metadata_dict = json.loads(attr_metadata)
    assert metadata_dict == metadata
    attr_tags = attributes.pop(TAG_TAGS, None)
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


@pytest.fixture(scope="module")
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(scope="module")
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> trace_api.TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    HTTPXClientInstrumentor().instrument(tracer_provider=tracer_provider)
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    OpenAIInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


@pytest.fixture(scope="module")
def seed() -> Iterator[int]:
    """
    Use rolling seeds to help debugging, because the rolling pseudo-random values
    allow conditional breakpoints to be hit precisely (and repeatably).
    """
    return count()


@pytest.fixture(autouse=True)
def set_seed(seed: Iterator[int]) -> Iterator[None]:
    random.seed(next(seed))
    yield


@pytest.fixture
def completion_usage() -> Dict[str, Any]:
    prompt_tokens = random.randint(1, 1000)
    completion_tokens = random.randint(1, 1000)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


@pytest.fixture
def model_name() -> str:
    return randstr()


@pytest.fixture
def input_messages() -> List[Dict[str, Any]]:
    return [{"role": randstr(), "content": randstr()} for _ in range(2)]


@pytest.fixture
def chat_completion_mock_stream() -> Tuple[List[bytes], List[Dict[str, Any]]]:
    return (
        [
            b'data: {"choices": [{"delta": {"role": "assistant"}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_amGrubFmr2FSPHeC5OPgwcNs", "function": {"arguments": "", "name": "get_current_weather"}, "type": "function"}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": ""}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"lo"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "{\\"lo"}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "catio"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "catio"}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "n\\": \\"B"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "n\\": \\"B"}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "osto"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "osto"}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "n, MA"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "n, MA"}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\\", \\"un"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "\\", \\"un"}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "it\\":"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "it\\":"}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": " \\"fah"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": " \\"fah"}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "renhei"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "renhei"}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "t\\"}"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "t\\"}"}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "call_6QTP4mLSYYzZwt3ZWj77vfZf", "function": {"arguments": "", "name": "get_current_weather"}, "type": "function"}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"role": "assistant"}, "index": 1}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "{\\"lo"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "{\\"lo"}, "index": 1}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "catio"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "catio"}, "index": 1}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "n\\": \\"S"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "n\\": \\"S"}, "index": 1}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "an F"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "an F"}, "index": 1}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "ranci"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "ranci"}, "index": 1}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "sco, C"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "sco, C"}, "index": 1}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "A\\", "}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "A\\", "}, "index": 1}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "\\"unit"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "\\"unit"}, "index": 1}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "\\": \\"fa"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "\\": \\"fa"}, "index": 1}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "hren"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "hren"}, "index": 1}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "heit\\""}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "heit\\""}, "index": 1}]}\n\n',
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "}"}}]}, "index": 0}]}\n\n',  # noqa: E501
            b'data: {"choices": [{"delta": {"content": "}"}, "index": 1}]}\n\n',
            b'data: {"choices": [{"finish_reason": "tool_calls", "index": 0}]}\n\n',  # noqa: E501
            b"data: [DONE]\n",
        ],
        [
            {
                "role": "assistant",
                "content": '{"location": "Boston, MA", "unit": "fahrenheit"}',
                "tool_calls": [
                    {
                        "id": "call_amGrubFmr2FSPHeC5OPgwcNs",
                        "function": {
                            "arguments": '{"location": "Boston, MA", "unit": "fahrenheit"}',
                            "name": "get_current_weather",
                        },
                        "type": "function",
                    },
                    {
                        "id": "call_6QTP4mLSYYzZwt3ZWj77vfZf",
                        "function": {
                            "arguments": '{"location": "San Francisco, CA", "unit": "fahrenheit"}',
                            "name": "get_current_weather",
                        },
                        "type": "function",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": '{"location": "San Francisco, CA", "unit": "fahrenheit"}',
            },
        ],
    )


@pytest.fixture
def completion_mock_stream() -> Tuple[List[bytes], List[str]]:
    return (
        [
            b'data: {"choices": [{"text": "", "index": 0}]}\n\n',
            b'data: {"choices": [{"text": "{\\"lo", "index": 1}]}\n\n',
            b'data: {"choices": [{"text": "{\\"lo", "index": 0}]}\n\n',
            b'data: {"choices": [{"text": "catio", "index": 1}]}\n\n',
            b'data: {"choices": [{"text": "catio", "index": 0}]}\n\n',
            b'data: {"choices": [{"text": "n\\": \\"S", "index": 1}]}\n\n',
            b'data: {"choices": [{"text": "n\\": \\"B", "index": 0}]}\n\n',
            b'data: {"choices": [{"text": "an F", "index": 1}]}\n\n',
            b'data: {"choices": [{"text": "osto", "index": 0}]}\n\n',
            b'data: {"choices": [{"text": "ranci", "index": 1}]}\n\n',
            b'data: {"choices": [{"text": "n, MA", "index": 0}]}\n\n',
            b'data: {"choices": [{"text": "sco, C", "index": 1}]}\n\n',
            b'data: {"choices": [{"text": "\\", \\"un", "index": 0}]}\n\n',
            b'data: {"choices": [{"text": "A\\", ", "index": 1}]}\n\n',
            b'data: {"choices": [{"text": "it\\":", "index": 0}]}\n\n',
            b'data: {"choices": [{"text": "\\"unit", "index": 1}]}\n\n',
            b'data: {"choices": [{"text": " \\"fah", "index": 0}]}\n\n',
            b'data: {"choices": [{"text": "\\": \\"fa", "index": 1}]}\n\n',
            b'data: {"choices": [{"text": "renhei", "index": 0}]}\n\n',
            b'data: {"choices": [{"text": "hren", "index": 1}]}\n\n',
            b'data: {"choices": [{"text": "t\\"}", "index": 0}]}\n\n',
            b'data: {"choices": [{"text": "heit\\"", "index": 1}]}\n\n',
            b'data: {"choices": [{"text": "}", "index": 1}]}\n\n',
            b"data: [DONE]\n",
        ],
        [
            '{"location": "Boston, MA", "unit": "fahrenheit"}',
            '{"location": "San Francisco, CA", "unit": "fahrenheit"}',
        ],
    )


class MockAsyncByteStream(AsyncByteStream):
    def __init__(self, byte_stream: Iterable[bytes]):
        self._byte_stream = byte_stream

    def __iter__(self) -> Iterator[bytes]:
        for byte_string in self._byte_stream:
            yield byte_string

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for byte_string in self._byte_stream:
            yield byte_string


def randstr() -> str:
    return str(random.random())


def get_texts() -> List[str]:
    return [randstr() for _ in range(2)]


def get_messages() -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        *[{"role": randstr(), "content": randstr()} for _ in range(2)],
        *[
            {
                "role": randstr(),
                "function_call": {"arguments": randstr(), "name": randstr()},
            }
            for _ in range(2)
        ],
        *(
            [
                {
                    "role": randstr(),
                    "tool_calls": [
                        {"function": {"arguments": randstr(), "name": randstr()}} for _ in range(2)
                    ],
                }
                for _ in range(2)
            ]
            if _openai_version() >= (1, 1, 0)
            else []
        ),
    ]
    random.shuffle(messages)
    return messages


def get_text_content() -> Dict[str, str]:
    return {
        "type": "text",
        "text": randstr(),
    }


def get_image_content() -> Dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": randstr(),
        },
    }


def get_messages_with_multiple_contents() -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": randstr(),
        },
        {
            "role": "user",
            "content": [
                get_text_content(),
                get_image_content(),
            ],
        },
    ]
    return messages


def _openai_version() -> Tuple[int, int, int]:
    return cast(Tuple[int, int, int], tuple(map(int, version("openai").split(".")[:3])))


def message_role(prefix: str, i: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_ROLE}"


def message_content(prefix: str, i: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENT}"


def message_contents_type(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENTS}.{j}.{MESSAGE_CONTENT_TYPE}"


def message_contents_text(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENTS}.{j}.{MESSAGE_CONTENT_TEXT}"


def message_contents_image_url(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENTS}.{j}.{MESSAGE_CONTENT_IMAGE}.{IMAGE_URL}"


def message_function_call_name(prefix: str, i: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_FUNCTION_CALL_NAME}"


def message_function_call_arguments(prefix: str, i: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON}"


def tool_call_function_name(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_TOOL_CALLS}.{j}.{TOOL_CALL_FUNCTION_NAME}"


def tool_call_function_arguments(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_TOOL_CALLS}.{j}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"


OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
IMAGE_URL = ImageAttributes.IMAGE_URL
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
EMBEDDING_EMBEDDINGS = SpanAttributes.EMBEDDING_EMBEDDINGS
EMBEDDING_MODEL_NAME = SpanAttributes.EMBEDDING_MODEL_NAME
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
SESSION_ID = SpanAttributes.SESSION_ID
USER_ID = SpanAttributes.USER_ID
METADATA = SpanAttributes.METADATA
TAG_TAGS = SpanAttributes.TAG_TAGS
