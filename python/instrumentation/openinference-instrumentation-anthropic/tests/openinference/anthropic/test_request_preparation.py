import json
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic, AsyncAnthropic, _base_client
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import TraceConfig, suppress_tracing, using_attributes
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from openinference.instrumentation.anthropic._wrappers import _Params
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)


def _response(request: httpx2.Request) -> httpx2.Response:
    message = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "body-model",
        "content": [{"type": "text", "text": "Hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }
    if json.loads(request.content).get("stream"):
        events: list[dict[str, Any]] = [
            {"type": "message_start", "message": {**message, "content": []}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "Hello"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
            ),
        )
    return httpx2.Response(200, json=message)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("is_beta", [False, True])
@pytest.mark.parametrize("is_stream", [False, True])
async def test_prepared_body_capture(
    is_async: bool,
    is_beta: bool,
    is_stream: bool,
    tmp_path: Path,
    setup_anthropic_instrumentation: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image data")
    requests = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return _response(request)

    messages: list[Any] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": image_path},
                },
            ],
        }
    ]
    kwargs: dict[str, Any] = {
        "model": "original-model",
        "max_tokens": 8,
        "messages": (message for message in messages),
        "extra_body": {"model": "body-model", "temperature": 0.25},
        "extra_query": {"model": "query-model"},
    }
    if is_beta:
        kwargs["messages"] = []
        kwargs["extra_body"]["messages"] = (message for message in messages)
    transport = httpx2.MockTransport(handler)
    result: Any
    with using_attributes(session_id="prepared-session"):
        if is_async:
            async with AsyncAnthropic(
                api_key="test", http_client=httpx2.AsyncClient(transport=transport)
            ) as async_client:
                if is_stream:
                    async_manager: AbstractAsyncContextManager[Any]
                    if is_beta:
                        async_manager = async_client.beta.messages.stream(**kwargs)
                    else:
                        async_manager = async_client.messages.stream(**kwargs)
                    async with async_manager as async_stream:
                        result = await async_stream.get_final_message()
                elif is_beta:
                    result = await async_client.beta.messages.create(**kwargs)
                else:
                    result = await async_client.messages.create(**kwargs)
        else:
            with Anthropic(
                api_key="test", http_client=httpx2.Client(transport=transport)
            ) as client:
                if is_stream:
                    manager: AbstractContextManager[Any]
                    if is_beta:
                        manager = client.beta.messages.stream(**kwargs)
                    else:
                        manager = client.messages.stream(**kwargs)
                    with manager as stream:
                        result = stream.get_final_message()
                elif is_beta:
                    result = client.beta.messages.create(**kwargs)
                else:
                    result = client.messages.create(**kwargs)
    assert result.model_dump()["content"][0]["text"] == "Hello"
    assert len(requests) == 1
    assert requests[0]["model"] == "body-model"
    assert requests[0]["temperature"] == 0.25
    assert requests[0]["messages"][0]["content"][0]["text"] == "Describe this"
    assert requests[0]["messages"][0]["content"][1]["source"]["data"] == "aW1hZ2UgZGF0YQ=="
    (span,) = in_memory_span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes[SpanAttributes.LLM_REQUEST_MODEL_NAME] == "body-model"
    invocation = json.loads(str(attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS]))
    assert invocation["temperature"] == 0.25
    assert attributes[SpanAttributes.SESSION_ID] == "prepared-session"
    content_prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}"
    assert attributes[f"{content_prefix}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"] == (
        "Describe this"
    )
    image_key = (
        f"{content_prefix}.1.{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}."
        f"{ImageAttributes.IMAGE_URL}"
    )
    assert attributes[image_key] == "data:image/png;base64,aW1hZ2UgZGF0YQ=="


@pytest.mark.parametrize("is_async", [False, True])
async def test_preparation_ignores_query_and_keeps_first_body(
    is_async: bool, setup_anthropic_instrumentation: None
) -> None:
    with _Params({"model": "original"}) as params:
        if is_async:
            query = await getattr(_base_client, "async_prepare_request_data")(
                {"model": "query-model"}, location="query"
            )
        else:
            query = getattr(_base_client, "prepare_request_data")(
                {"model": "query-model"}, location="query"
            )
        assert query == {"model": "query-model"}
        assert dict(params) == {"model": "original"}
        for model in ("first-body", "second-body"):
            if is_async:
                body = await getattr(_base_client, "async_prepare_request_data")(
                    {"model": model}, location="body"
                )
            else:
                body = getattr(_base_client, "prepare_request_data")(
                    {"model": model}, location="body"
                )
            assert body == {"model": model}
        assert dict(params) == {"model": "first-body"}


def test_preparation_lifecycle_suppression_and_masking(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    instrumentor = AnthropicInstrumentor()
    original_prepare = getattr(_base_client, "prepare_request_data")
    original_async_prepare = getattr(_base_client, "async_prepare_request_data")
    with Anthropic(
        api_key="test", http_client=httpx2.Client(transport=httpx2.MockTransport(_response))
    ) as client:
        for _ in range(2):
            instrumentor.instrument(
                tracer_provider=tracer_provider, config=TraceConfig(hide_inputs=True)
            )
            try:
                with suppress_tracing():
                    client.messages.create(
                        model="body-model",
                        max_tokens=8,
                        messages=[{"role": "user", "content": "secret"}],
                    )
                assert not in_memory_span_exporter.get_finished_spans()
                client.messages.create(
                    model="body-model",
                    max_tokens=8,
                    messages=[{"role": "user", "content": "secret"}],
                )
                (span,) = in_memory_span_exporter.get_finished_spans()
                attributes = dict(span.attributes or {})
                assert attributes[SpanAttributes.INPUT_VALUE] == "__REDACTED__"
                assert not any(
                    key.startswith(SpanAttributes.LLM_INPUT_MESSAGES) for key in attributes
                )
            finally:
                instrumentor.uninstrument()
            assert getattr(_base_client, "prepare_request_data") is original_prepare
            assert getattr(_base_client, "async_prepare_request_data") is original_async_prepare
            in_memory_span_exporter.clear()
            response = client.messages.create(
                model="body-model", max_tokens=8, messages=[{"role": "user", "content": "untraced"}]
            )
            assert response.model_dump()["content"][0]["text"] == "Hello"
            assert not in_memory_span_exporter.get_finished_spans()


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("is_beta", [False, True])
async def test_stream_preparation_failure_finishes_span(
    is_async: bool,
    is_beta: bool,
    tmp_path: Path,
    setup_anthropic_instrumentation: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    missing_image = tmp_path / "missing.png"
    kwargs: dict[str, Any] = {
        "model": "body-model",
        "max_tokens": 8,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": missing_image,
                        },
                    }
                ],
            }
        ],
    }
    transport = httpx2.MockTransport(_response)
    with pytest.raises(FileNotFoundError) as error:
        if is_async:
            async with AsyncAnthropic(
                api_key="test", http_client=httpx2.AsyncClient(transport=transport)
            ) as async_client:
                async_manager: AbstractAsyncContextManager[Any]
                if is_beta:
                    async_manager = async_client.beta.messages.stream(**kwargs)
                else:
                    async_manager = async_client.messages.stream(**kwargs)
                async with async_manager:
                    pytest.fail("Request preparation must fail before stream entry")
        else:
            with Anthropic(
                api_key="test", http_client=httpx2.Client(transport=transport)
            ) as client:
                manager: AbstractContextManager[Any]
                if is_beta:
                    manager = client.beta.messages.stream(**kwargs)
                else:
                    manager = client.messages.stream(**kwargs)
                with manager:
                    pytest.fail("Request preparation must fail before stream entry")
    assert error.value.filename == str(missing_image)
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code.name == "ERROR"
    assert span.events[0].name == "exception"
    assert (span.events[0].attributes or {})["exception.type"] == "FileNotFoundError"
