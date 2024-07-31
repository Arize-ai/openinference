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
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import pytest
from httpx import AsyncByteStream, Response
from openinference.instrumentation import REDACTED_VALUE, TraceConfig, using_attributes
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
            if _openai_version() >= (1, 6, 0):
                async with response as iterator:
                    async for _ in iterator:
                        pass
            else:
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
                        if _openai_version() >= (1, 6, 0):
                            with response as iterator:
                                for _ in iterator:
                                    pass
                        else:
                            for _ in response:
                                pass
        else:
            if is_async:
                asyncio.run(task())
            else:
                response = create(**create_kwargs)
                response = response.parse() if is_raw else response
                if is_stream:
                    if _openai_version() >= (1, 6, 0):
                        with response as iterator:
                            for _ in iterator:
                                pass
                    else:
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
    for i, message in enumerate(input_messages):
        _check_llm_message(LLM_INPUT_MESSAGES, i, attributes, message)

    if status_code == 200:
        for i, message in enumerate(output_messages):
            _check_llm_message(LLM_OUTPUT_MESSAGES, i, attributes, message)

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
    for i, message in enumerate(input_messages):
        _check_llm_message(LLM_INPUT_MESSAGES, i, attributes, message)

    if status_code == 200:
        for i, message in enumerate(output_messages):
            _check_llm_message(LLM_OUTPUT_MESSAGES, i, attributes, message)

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


@pytest.mark.parametrize("hide_inputs", [False, True])
@pytest.mark.parametrize("hide_input_messages", [False, True])
@pytest.mark.parametrize("hide_input_images", [False, True])
@pytest.mark.parametrize("hide_input_text", [False, True])
@pytest.mark.parametrize("base64_image_max_length", [0, 100_000])
def test_chat_completions_with_config_hiding_hiding_inputs(
    tracer_provider: trace_api.TracerProvider,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    completion_usage: Dict[str, Any],
    model_name: str,
    hide_inputs: bool,
    hide_input_messages: bool,
    hide_input_images: bool,
    hide_input_text: bool,
    base64_image_max_length: int,
) -> None:
    OpenAIInstrumentor().uninstrument()
    config = TraceConfig(
        hide_inputs=hide_inputs,
        hide_input_messages=hide_input_messages,
        hide_input_images=hide_input_images,
        hide_input_text=hide_input_text,
        base64_image_max_length=base64_image_max_length,
    )
    assert config.hide_inputs is hide_inputs
    assert config.hide_input_messages is hide_input_messages
    assert config.hide_input_images is hide_input_images
    assert config.hide_input_text is hide_input_text
    assert config.base64_image_max_length is base64_image_max_length
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, config=config)
    input_messages: List[Dict[str, Any]] = get_messages_with_multiple_contents()
    output_messages: List[Dict[str, Any]] = get_messages()
    invocation_parameters = {
        "model": model_name,
        "temperature": random.random(),
        "n": len(output_messages),
    }
    url = "https://api.openai.com/v1/chat/completions"
    respx_kwargs: Dict[str, Any] = {
        **(
            {
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
    respx_mock.post(url).mock(return_value=Response(status_code=200, **respx_kwargs))
    create_kwargs = {"messages": input_messages, **invocation_parameters}
    openai = import_module("openai")
    completions = openai.OpenAI(api_key="sk-").chat.completions
    create = completions.create

    with suppress(openai.BadRequestError):
        _ = create(**create_kwargs)
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2  # first span should be from the httpx instrumentor
    span: ReadableSpan = spans[1]
    assert span.status.is_ok
    assert not span.status.description
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
    if not hide_inputs and not hide_input_messages:
        for i, message in enumerate(input_messages):
            _check_llm_message(
                LLM_INPUT_MESSAGES,
                i,
                attributes,
                message,
                hide_text=hide_input_text,
                hide_images=hide_input_images,
                image_limit=base64_image_max_length,
            )

    for i, message in enumerate(output_messages):
        _check_llm_message(LLM_OUTPUT_MESSAGES, i, attributes, message)

    assert isinstance(attributes.pop(OUTPUT_VALUE, None), str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE, None))
        == OpenInferenceMimeTypeValues.JSON
    )
    # Usage is not available for streaming in general.
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL, None) == completion_usage["total_tokens"]
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT, None) == completion_usage["prompt_tokens"]
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION, None) == completion_usage["completion_tokens"]
    # We left out model_name from our mock stream.
    assert attributes.pop(LLM_MODEL_NAME, None) == model_name
    assert attributes == {}  # test should account for all span attributes
    OpenAIInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


@pytest.mark.parametrize("hide_outputs", [False, True])
@pytest.mark.parametrize("hide_output_messages", [False, True])
@pytest.mark.parametrize("hide_output_text", [False, True])
def test_chat_completions_with_config_hiding_hiding_outputs(
    tracer_provider: trace_api.TracerProvider,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    completion_usage: Dict[str, Any],
    model_name: str,
    hide_outputs: bool,
    hide_output_messages: bool,
    hide_output_text: bool,
) -> None:
    OpenAIInstrumentor().uninstrument()
    config = TraceConfig(
        hide_outputs=hide_outputs,
        hide_output_messages=hide_output_messages,
        hide_output_text=hide_output_text,
    )
    assert config.hide_outputs is hide_outputs
    assert config.hide_output_messages is hide_output_messages
    assert config.hide_output_text is hide_output_text
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, config=config)
    input_messages: List[Dict[str, Any]] = get_messages_with_multiple_contents()
    output_messages: List[Dict[str, Any]] = get_messages()
    invocation_parameters = {
        "model": model_name,
        "temperature": random.random(),
        "n": len(output_messages),
    }
    url = "https://api.openai.com/v1/chat/completions"
    respx_kwargs: Dict[str, Any] = {
        **(
            {
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
    respx_mock.post(url).mock(return_value=Response(status_code=200, **respx_kwargs))
    create_kwargs = {"messages": input_messages, **invocation_parameters}
    openai = import_module("openai")
    completions = openai.OpenAI(api_key="sk-").chat.completions
    create = completions.create

    with suppress(openai.BadRequestError):
        _ = create(**create_kwargs)
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2  # first span should be from the httpx instrumentor
    span: ReadableSpan = spans[1]
    assert span.status.is_ok
    assert not span.status.description
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
    for i, message in enumerate(input_messages):
        _check_llm_message(
            LLM_INPUT_MESSAGES,
            i,
            attributes,
            message,
        )

    if not hide_outputs and not hide_output_messages:
        for i, message in enumerate(output_messages):
            _check_llm_message(
                LLM_OUTPUT_MESSAGES,
                i,
                attributes,
                message,
                hide_text=hide_output_text,
            )

    output_value = attributes.pop(OUTPUT_VALUE, None)
    assert output_value is not None
    if hide_outputs:
        output_value == REDACTED_VALUE
    else:
        assert isinstance(output_value, str)
        assert (
            OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE, None))
            == OpenInferenceMimeTypeValues.JSON
        )
    # Usage is not available for streaming in general.
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL, None) == completion_usage["total_tokens"]
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT, None) == completion_usage["prompt_tokens"]
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION, None) == completion_usage["completion_tokens"]
    # We left out model_name from our mock stream.
    assert attributes.pop(LLM_MODEL_NAME, None) == model_name
    assert attributes == {}  # test should account for all span attributes
    OpenAIInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


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

    if function_call := message.get("function_call"):
        assert attributes.pop(message_function_call_name(prefix, i), None) == function_call.get(
            "name"
        )
        assert attributes.pop(
            message_function_call_arguments(prefix, i), None
        ) == function_call.get("arguments")
    if _openai_version() >= (1, 1, 0) and (tool_calls := message.get("tool_calls")):
        for j, tool_call in enumerate(tool_calls):
            if function := tool_call.get("function"):
                assert attributes.pop(tool_call_function_name(prefix, i, j), None) == function.get(
                    "name"
                )
                assert attributes.pop(
                    tool_call_function_arguments(prefix, i, j), None
                ) == function.get("arguments")


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
            "url": get_base64_img_url(),
        },
    }


def get_base64_img_url() -> str:
    base64_str = "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"  # noqa: E501
    return f"data:image/jpeg;base64,{base64_str}"


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
