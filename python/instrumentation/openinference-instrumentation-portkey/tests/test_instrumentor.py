import gc
import json
from importlib import import_module
from typing import Any, Dict, Iterator, Tuple

import httpx
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
    attributes = dict(spans[0].attributes or {})
    assert (
        attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.LLM.value
    )
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o-mini"
    assert attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS)
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    input_value = attributes.pop(SpanAttributes.INPUT_VALUE)
    assert isinstance(input_value, str)
    assert json.loads(input_value)["messages"][0]["content"] == list(parts)
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE)
    assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "stop"

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
    assert attributes.pop(f"{output_message}.{MessageAttributes.MESSAGE_CONTENT}") == "A cat."
    assert not attributes


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


_STREAM_CHUNKS: Tuple[Dict[str, Any], ...] = (
    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]},
    {
        "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
    },
)


def _stream_events(chunks: Tuple[Dict[str, Any], ...]) -> str:
    header = {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "created": 1750000000,
        "model": "gpt-4o-mini",
    }
    return "".join(f"data: {json.dumps({**header, **chunk})}\n\n" for chunk in chunks)


def _stream_response(chunks: Tuple[Dict[str, Any], ...] = _STREAM_CHUNKS) -> Response:
    """A chat-completion SSE stream, read by both the portkey_ai 1.x and 2.x clients."""
    return Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content=(_stream_events(chunks) + "data: [DONE]\n\n").encode(),
    )


def _pop_stream_span(
    in_memory_span_exporter: InMemorySpanExporter,
    name: str,
    status_code: trace_api.StatusCode,
) -> Dict[str, Any]:
    """Checks the span of a streamed call and returns its attributes, less the request's."""
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == name
    assert span.status.status_code == status_code
    attributes: Dict[str, Any] = dict(span.attributes or {})
    assert (
        attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.LLM.value
    )
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert json.loads(attributes.pop(SpanAttributes.INPUT_VALUE)) == {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    assert json.loads(attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS)) == {
        "model": "gpt-4o-mini",
        "stream": True,
    }
    message = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0"
    assert attributes.pop(f"{message}.{MessageAttributes.MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{message}.{MessageAttributes.MESSAGE_CONTENT}") == "hi"
    return attributes


def _assert_stream_span(in_memory_span_exporter: InMemorySpanExporter, name: str) -> None:
    attributes = _pop_stream_span(in_memory_span_exporter, name, trace_api.StatusCode.OK)
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "Hello"
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o-mini"
    assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "stop"
    output_message = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    assert attributes.pop(f"{output_message}.{MessageAttributes.MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(f"{output_message}.{MessageAttributes.MESSAGE_CONTENT}") == "Hello"
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 8
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 2
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 10
    assert not attributes


def _assert_truncated_stream_span(in_memory_span_exporter: InMemorySpanExporter, name: str) -> None:
    """The span of a stream that was stopped after its first chunk."""
    attributes = _pop_stream_span(in_memory_span_exporter, name, trace_api.StatusCode.UNSET)
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "Hel"
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o-mini"
    output_message = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    assert attributes.pop(f"{output_message}.{MessageAttributes.MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(f"{output_message}.{MessageAttributes.MESSAGE_CONTENT}") == "Hel"
    assert not attributes


def _create_stream(**kwargs: Any) -> Any:
    portkey = import_module("portkey_ai")
    client = portkey.Portkey(api_key="REDACTED", virtual_key="REDACTED")
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        **kwargs,
    )


def test_chat_completion_stream(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()
    portkey = import_module("portkey_ai")
    client = portkey.Portkey(api_key="REDACTED", virtual_key="REDACTED")
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(return_value=_stream_response())
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        # The span stays open until the stream is read.
        assert not in_memory_span_exporter.get_finished_spans()
        text = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
    assert text == "Hello"
    _assert_stream_span(in_memory_span_exporter, "Completions")


@pytest.mark.asyncio
async def test_async_chat_completion_stream(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()
    portkey = import_module("portkey_ai")
    client = portkey.AsyncPortkey(api_key="REDACTED", virtual_key="REDACTED")
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(return_value=_stream_response())
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        assert not in_memory_span_exporter.get_finished_spans()
        text = "".join(
            [chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices]
        )
    assert text == "Hello"
    _assert_stream_span(in_memory_span_exporter, "AsyncCompletions")


def test_chat_completion_stream_closed_early(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(return_value=_stream_response())
        stream = _create_stream()
        next(stream)
        stream.close()
    _assert_truncated_stream_span(in_memory_span_exporter, "Completions")


@pytest.mark.asyncio
async def test_async_chat_completion_stream_closed_early(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()
    portkey = import_module("portkey_ai")
    client = portkey.AsyncPortkey(api_key="REDACTED", virtual_key="REDACTED")
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(return_value=_stream_response())
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        await stream.__anext__()
        if hasattr(stream.__wrapped__, "aclose"):
            # portkey_ai 1.x: an async generator
            await stream.aclose()
        else:
            # portkey_ai 2.x: openai's AsyncStream
            await stream.close()
    _assert_truncated_stream_span(in_memory_span_exporter, "AsyncCompletions")


def test_chat_completion_stream_context_manager_exited_early(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(return_value=_stream_response())
        stream = _create_stream()
        if not hasattr(stream.__wrapped__, "__enter__"):
            stream.close()
            pytest.skip("portkey_ai 1.x streams through a generator, not a context manager")
        with stream:
            for _ in stream:
                break
    _assert_truncated_stream_span(in_memory_span_exporter, "Completions")


def test_chat_completion_stream_abandoned(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()
    # portkey_ai 1.x sends the request on the first read, so it may never be called.
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=False) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(return_value=_stream_response())
        stream = _create_stream()
        assert not in_memory_span_exporter.get_finished_spans()
        del stream
        gc.collect()
    attributes = _pop_stream_span(
        in_memory_span_exporter, "Completions", trace_api.StatusCode.UNSET
    )
    assert not attributes


def test_chat_completion_stream_error(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()

    def body() -> Iterator[bytes]:
        yield _stream_events(_STREAM_CHUNKS[:1]).encode()
        raise httpx.ReadError("connection lost")

    def respond(request: httpx.Request) -> Response:
        return Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=body(),
        )

    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(side_effect=respond)
        stream = _create_stream()
        with pytest.raises(httpx.ReadError, match="connection lost"):
            for _ in stream:
                pass
    events = in_memory_span_exporter.get_finished_spans()[0].events
    assert [event.name for event in events] == ["exception"]
    attributes = _pop_stream_span(
        in_memory_span_exporter, "Completions", trace_api.StatusCode.ERROR
    )
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "Hel"
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o-mini"
    output_message = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    assert attributes.pop(f"{output_message}.{MessageAttributes.MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(f"{output_message}.{MessageAttributes.MESSAGE_CONTENT}") == "Hel"
    assert not attributes


def test_chat_completion_stream_tool_calls(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()
    chunks: Tuple[Dict[str, Any], ...] = (
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            },
                        ],
                    },
                },
            ],
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]},
                },
            ],
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]},
                    "finish_reason": "tool_calls",
                },
            ],
        },
    )
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(return_value=_stream_response(chunks))
        for _ in _create_stream():
            pass
    attributes = _pop_stream_span(in_memory_span_exporter, "Completions", trace_api.StatusCode.OK)
    assert json.loads(attributes.pop(SpanAttributes.OUTPUT_VALUE)) == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                },
            ],
        },
    ]
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o-mini"
    assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "tool_calls"
    output_message = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    assert attributes.pop(f"{output_message}.{MessageAttributes.MESSAGE_ROLE}") == "assistant"
    assert not attributes


def test_prompt_completion_stream(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()
    chunks: Tuple[Dict[str, Any], ...] = (
        {"choices": [{"index": 0, "text": "Hel"}]},
        {"choices": [{"index": 0, "text": "lo", "finish_reason": "stop"}]},
    )
    portkey = import_module("portkey_ai")
    client = portkey.Portkey(api_key="REDACTED", virtual_key="REDACTED")
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/prompts/pp-test/completions").mock(
            return_value=_stream_response(chunks)
        )
        for _ in client.prompts.completions.create(prompt_id="pp-test", stream=True):
            pass
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == trace_api.StatusCode.OK
    attributes: Dict[str, Any] = dict(spans[0].attributes or {})
    assert (
        attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.LLM.value
    )
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert json.loads(attributes.pop(SpanAttributes.INPUT_VALUE)) == {
        "prompt_id": "pp-test",
        "stream": True,
    }
    assert json.loads(attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS)) == {
        "prompt_id": "pp-test",
        "stream": True,
    }
    assert attributes.pop(SpanAttributes.PROMPT_ID) == "pp-test"
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "Hello"
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o-mini"
    assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "stop"
    output_message = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    assert attributes.pop(f"{output_message}.{MessageAttributes.MESSAGE_CONTENT}") == "Hello"
    assert not attributes
