import json
from importlib import import_module
from typing import Any, Iterator

import pytest
import respx
from httpx import Response
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.portkey._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.semconv.trace import (
    AudioAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


@pytest.mark.vcr(
    before_record_request=lambda request: setattr(  # type: ignore[func-returns-value]
        request,
        "headers",
        {k: v for k, v in request.headers.items() if not k.lower().startswith("x-portkey")},
    )
    or request,
    before_record_response=lambda response: {
        **response,
        "headers": {
            k: v for k, v in response["headers"].items() if not k.lower().startswith("x-portkey")
        },
    },
)
def test_chat_completion(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    portkey = import_module("portkey_ai")
    client = portkey.Portkey(
        api_key="REDACTED",
        virtual_key="REDACTED",
    )
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": "What's the weather like?"}], model="gpt-4o-mini"
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})

    expected_attributes = {
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0."
        f"{MessageAttributes.MESSAGE_CONTENT}": "What's the weather like?",
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gpt-4o-mini-2024-07-18",
        SpanAttributes.LLM_FINISH_REASON: resp.choices[0].finish_reason,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: resp.usage.total_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: resp.usage.prompt_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: resp.usage.completion_tokens,
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": resp.choices[
            0
        ].message.role,
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": resp.choices[
            0
        ].message.content,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    for key, expected_value in expected_attributes.items():
        assert attributes.get(key) == expected_value


@pytest.mark.vcr(
    before_record_request=lambda request: setattr(  # type: ignore[func-returns-value]
        request,
        "headers",
        {k: v for k, v in request.headers.items() if not k.lower().startswith("x-portkey")},
    )
    or request,
    before_record_response=lambda response: {
        **response,
        "headers": {
            k: v for k, v in response["headers"].items() if not k.lower().startswith("x-portkey")
        },
    },
)
def test_prompt_template(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    prompt_id = "pp-weather-pr-b74c4f"
    portkey = import_module("portkey_ai")
    variables = {"location": "New York City"}
    client = portkey.Portkey(
        api_key="REDACTED",
        virtual_key="REDACTED",
    )
    resp = client.prompts.completions.create(
        prompt_id=prompt_id,
        variables=variables,
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})

    expected_attributes = {
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gpt-4.1-2025-04-14",
        SpanAttributes.LLM_FINISH_REASON: resp.choices[0].finish_reason,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: resp.usage.total_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: resp.usage.prompt_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: resp.usage.completion_tokens,
        SpanAttributes.PROMPT_ID: prompt_id,
        SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES: json.dumps(variables),
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "assistant",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": resp.choices[
            0
        ].message.content,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    for key, expected_value in expected_attributes.items():
        assert attributes.get(key) == expected_value


@pytest.mark.asyncio
@pytest.mark.vcr(
    before_record_request=lambda request: setattr(  # type: ignore[func-returns-value]
        request,
        "headers",
        {k: v for k, v in request.headers.items() if not k.lower().startswith("x-portkey")},
    )
    or request,
    before_record_response=lambda response: {
        **response,
        "headers": {
            k: v for k, v in response["headers"].items() if not k.lower().startswith("x-portkey")
        },
    },
)
async def test_async_chat_completion(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    portkey = import_module("portkey_ai")
    client = portkey.AsyncPortkey(
        api_key="REDACTED",
        virtual_key="REDACTED",
    )
    resp = await client.chat.completions.create(
        messages=[{"role": "user", "content": "What's the weather like?"}], model="gpt-4o-mini"
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "AsyncCompletions"
    attributes = dict(span.attributes or {})

    expected_attributes = {
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0."
        f"{MessageAttributes.MESSAGE_CONTENT}": "What's the weather like?",
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gpt-4o-mini-2024-07-18",
        SpanAttributes.LLM_FINISH_REASON: resp.choices[0].finish_reason,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: resp.usage.total_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: resp.usage.prompt_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: resp.usage.completion_tokens,
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": resp.choices[
            0
        ].message.role,
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": resp.choices[
            0
        ].message.content,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    for key, expected_value in expected_attributes.items():
        assert attributes.get(key) == expected_value


@pytest.mark.asyncio
@pytest.mark.vcr(
    before_record_request=lambda request: setattr(  # type: ignore[func-returns-value]
        request,
        "headers",
        {k: v for k, v in request.headers.items() if not k.lower().startswith("x-portkey")},
    )
    or request,
    before_record_response=lambda response: {
        **response,
        "headers": {
            k: v for k, v in response["headers"].items() if not k.lower().startswith("x-portkey")
        },
    },
)
async def test_async_prompt_template(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    prompt_id = "pp-weather-pr-b74c4f"
    portkey = import_module("portkey_ai")
    variables = {"location": "New York City"}
    client = portkey.AsyncPortkey(
        api_key="REDACTED",
        virtual_key="REDACTED",
    )
    resp = await client.prompts.completions.create(
        prompt_id=prompt_id,
        variables=variables,
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "AsyncCompletions"
    attributes = dict(span.attributes or {})

    expected_attributes = {
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gpt-4.1-2025-04-14",
        SpanAttributes.LLM_FINISH_REASON: resp.choices[0].finish_reason,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: resp.usage.total_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: resp.usage.prompt_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: resp.usage.completion_tokens,
        SpanAttributes.PROMPT_ID: prompt_id,
        SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES: json.dumps(variables),
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "assistant",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": resp.choices[
            0
        ].message.content,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    for key, expected_value in expected_attributes.items():
        assert attributes.get(key) == expected_value


@pytest.mark.parametrize(
    "finish_reason",
    [
        "stop",
        "length",
        "tool_calls",
        "content_filter",
    ],
)
def test_finish_reason_values(
    finish_reason: str,
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()

    with respx.mock(
        base_url="https://api.portkey.ai",
        assert_all_called=True,
    ) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(
            return_value=Response(
                status_code=200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1750000000,
                    "model": "gpt-4o-mini-2024-07-18",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello!",
                            },
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 10,
                        "total_tokens": 15,
                    },
                },
            )
        )

        portkey = import_module("portkey_ai")
        client = portkey.Portkey(
            api_key="REDACTED",
            virtual_key="REDACTED",
        )
        client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})
    assert attributes.get(SpanAttributes.LLM_FINISH_REASON) == finish_reason


def test_uninstrument_restores_all_wrapped_methods(
    tracer_provider: trace_api.TracerProvider,
) -> None:
    from portkey_ai.api_resources.apis import chat_complete, generation

    from openinference.instrumentation.portkey import PortkeyInstrumentor

    instrumentor = PortkeyInstrumentor()
    original_chat_create = chat_complete.Completions.create
    original_async_chat_create = chat_complete.AsyncCompletions.create
    original_prompt_create = generation.Completions.create
    original_async_prompt_create = generation.AsyncCompletions.create

    try:
        instrumentor.instrument(tracer_provider=tracer_provider)

        assert chat_complete.Completions.create is not original_chat_create
        assert chat_complete.AsyncCompletions.create is not original_async_chat_create
        assert generation.Completions.create is not original_prompt_create
        assert generation.AsyncCompletions.create is not original_async_prompt_create
    finally:
        instrumentor.uninstrument()

    assert chat_complete.Completions.create is original_chat_create
    assert chat_complete.AsyncCompletions.create is original_async_chat_create
    assert generation.Completions.create is original_prompt_create
    assert generation.AsyncCompletions.create is original_async_prompt_create


def test_chat_completion_with_multimodal_input(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()

    with respx.mock(
        base_url="https://api.portkey.ai",
        assert_all_called=True,
    ) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(
            return_value=Response(
                status_code=200,
                json={
                    "id": "chatcmpl-multimodal",
                    "object": "chat.completion",
                    "created": 1750000000,
                    "model": "gpt-4o-mini-2024-07-18",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "A cat sitting on a couch.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 8,
                        "total_tokens": 28,
                    },
                },
            )
        )

        portkey = import_module("portkey_ai")
        client = portkey.Portkey(
            api_key="REDACTED",
            virtual_key="REDACTED",
        )
        client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/cat.png"},
                        },
                    ],
                }
            ],
            model="gpt-4o-mini",
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    assert (
        attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.LLM.value
    )
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o-mini-2024-07-18"
    assert attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS)
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(SpanAttributes.INPUT_VALUE)
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE)
    assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "stop"
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 20
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 8
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 28

    message = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0"
    assert attributes.pop(f"{message}.{MessageAttributes.MESSAGE_ROLE}") == "user"
    contents = f"{message}.{MessageAttributes.MESSAGE_CONTENTS}"
    assert attributes.pop(f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}") == "text"
    assert (
        attributes.pop(f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}")
        == "What is in this image?"
    )
    assert (
        attributes.pop(f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}") == "image"
    )
    assert (
        attributes.pop(
            f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}."
            f"{ImageAttributes.IMAGE_URL}"
        )
        == "https://example.com/cat.png"
    )

    output_message = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    assert attributes.pop(f"{output_message}.{MessageAttributes.MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{output_message}.{MessageAttributes.MESSAGE_CONTENT}")
        == "A cat sitting on a couch."
    )
    assert not attributes


def test_chat_completion_with_input_audio(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()

    with respx.mock(
        base_url="https://api.portkey.ai",
        assert_all_called=True,
    ) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(
            return_value=Response(
                status_code=200,
                json={
                    "id": "chatcmpl-multimodal-audio",
                    "object": "chat.completion",
                    "created": 1750000000,
                    "model": "gpt-4o-mini-2024-07-18",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "The audio says meow.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 8,
                        "total_tokens": 28,
                    },
                },
            )
        )

        portkey = import_module("portkey_ai")
        client = portkey.Portkey(
            api_key="REDACTED",
            virtual_key="REDACTED",
        )
        client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What does this audio say?"},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "ZmFrZQ==", "format": "mp3"},
                        },
                    ],
                }
            ],
            model="gpt-4o-mini",
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    assert (
        attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.LLM.value
    )
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o-mini-2024-07-18"
    assert attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS)
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(SpanAttributes.INPUT_VALUE)
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE)
    assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "stop"
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 20
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 8
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 28

    message = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0"
    assert attributes.pop(f"{message}.{MessageAttributes.MESSAGE_ROLE}") == "user"
    contents = f"{message}.{MessageAttributes.MESSAGE_CONTENTS}"
    assert attributes.pop(f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}") == "text"
    assert (
        attributes.pop(f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}")
        == "What does this audio say?"
    )
    assert (
        attributes.pop(f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}") == "audio"
    )
    assert (
        attributes.pop(
            f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_AUDIO}."
            f"{AudioAttributes.AUDIO_URL}"
        )
        == "data:audio/mpeg;base64,ZmFrZQ=="
    )

    output_message = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    assert attributes.pop(f"{output_message}.{MessageAttributes.MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{output_message}.{MessageAttributes.MESSAGE_CONTENT}")
        == "The audio says meow."
    )
    assert not attributes


def test_message_content_as_tuple() -> None:
    # The SDKs accept any iterable of content parts, not only lists.
    message = {
        "role": "user",
        "content": (
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ),
    }
    attributes = dict(_RequestAttributesExtractor()._get_attributes_from_message_param(message))
    assert attributes.pop(MessageAttributes.MESSAGE_ROLE) == "user"
    contents = MessageAttributes.MESSAGE_CONTENTS
    assert attributes.pop(f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}") == "text"
    assert (
        attributes.pop(f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}")
        == "What is in this image?"
    )
    assert (
        attributes.pop(f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}") == "image"
    )
    assert (
        attributes.pop(
            f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}."
            f"{ImageAttributes.IMAGE_URL}"
        )
        == "https://example.com/cat.png"
    )
    assert not attributes


def test_chat_completion_with_generator_content(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()
    parts = (
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
    )
    content: Iterator[Any] = (part for part in parts)

    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        route = respx_mock.post("/v1/chat/completions").mock(
            return_value=Response(
                status_code=200,
                json={
                    "id": "chatcmpl-generator",
                    "object": "chat.completion",
                    "created": 1750000000,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "A cat."},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )
        portkey = import_module("portkey_ai")
        client = portkey.Portkey(api_key="REDACTED", virtual_key="REDACTED")
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
        )

    # The generator is read once for the span, and the SDK still sends every part.
    assert json.loads(route.calls.last.request.content)["messages"][0]["content"] == list(parts)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    input_messages = {
        key: value
        for key, value in (spans[0].attributes or {}).items()
        if key.startswith(SpanAttributes.LLM_INPUT_MESSAGES)
    }
    message = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0"
    assert input_messages.pop(f"{message}.{MessageAttributes.MESSAGE_ROLE}") == "user"
    contents = f"{message}.{MessageAttributes.MESSAGE_CONTENTS}"
    assert (
        input_messages.pop(f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert (
        input_messages.pop(f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}")
        == "What is in this image?"
    )
    assert (
        input_messages.pop(f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}")
        == "image"
    )
    assert (
        input_messages.pop(
            f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}."
            f"{ImageAttributes.IMAGE_URL}"
        )
        == "https://example.com/cat.png"
    )
    assert not input_messages


def _failing_content() -> Iterator[Any]:
    yield {"type": "text", "text": "What is in this image?"}
    raise RuntimeError("content failed")


def _assert_failed_content_span(in_memory_span_exporter: InMemorySpanExporter) -> None:
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == trace_api.StatusCode.ERROR
    assert span.status.description == "RuntimeError: content failed"
    assert [event.name for event in span.events] == ["exception"]


def test_chat_completion_with_failing_generator_content(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()
    portkey = import_module("portkey_ai")
    client = portkey.Portkey(api_key="REDACTED", virtual_key="REDACTED")
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=False) as respx_mock:
        route = respx_mock.post("/v1/chat/completions")
        with pytest.raises(RuntimeError, match="content failed"):
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": _failing_content()}],
            )
    assert not route.called
    _assert_failed_content_span(in_memory_span_exporter)


@pytest.mark.asyncio
async def test_async_chat_completion_with_failing_generator_content(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()
    portkey = import_module("portkey_ai")
    client = portkey.AsyncPortkey(api_key="REDACTED", virtual_key="REDACTED")
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=False) as respx_mock:
        route = respx_mock.post("/v1/chat/completions")
        with pytest.raises(RuntimeError, match="content failed"):
            await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": _failing_content()}],
            )
    assert not route.called
    _assert_failed_content_span(in_memory_span_exporter)
