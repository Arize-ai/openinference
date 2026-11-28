import json
from typing import Any, Dict

import httpx
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from together import APIResponse, AsyncTogether, Together

from openinference.instrumentation.together._wrappers import _is_parseable
from openinference.semconv.trace import MessageAttributes, SpanAttributes

COMPLETION: Dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "llama-3.3-70b-versatile",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Paris."},
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}

CHUNK: Dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion.chunk",
    "created": 1,
    "model": "llama-3.3-70b-versatile",
    "choices": [
        {"index": 0, "delta": {"role": "assistant", "content": "Paris."}, "finish_reason": "stop"}
    ],
}

MESSAGES: Any = [{"role": "user", "content": "What is the capital of France?"}]


def _handler(request: httpx.Request) -> httpx.Response:
    if json.loads(request.content).get("stream"):
        body = f"data: {json.dumps(CHUNK)}\n\ndata: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=COMPLETION)


def _assert_completion_attributes(exporter: InMemorySpanExporter) -> None:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})
    assert span.status.status_code == StatusCode.OK
    assert attributes[SpanAttributes.LLM_MODEL_NAME] == "llama-3.3-70b-versatile"
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 10
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 2
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 12
    output = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    assert attributes[f"{output}.{MessageAttributes.MESSAGE_ROLE}"] == "assistant"
    assert attributes[f"{output}.{MessageAttributes.MESSAGE_CONTENT}"] == "Paris."
    assert json.loads(str(attributes[SpanAttributes.OUTPUT_VALUE]))["id"] == "chatcmpl-1"


def test_with_raw_response(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    client = Together(
        api_key="fake", http_client=httpx.Client(transport=httpx.MockTransport(_handler))
    )
    response = client.chat.completions.with_raw_response.create(
        model="llama-3.3-70b-versatile", messages=MESSAGES
    )

    # The caller still gets the raw response and can parse it
    assert isinstance(response, APIResponse)
    message = response.parse().choices[0].message
    assert message is not None and message.content == "Paris."
    _assert_completion_attributes(in_memory_span_exporter)


@pytest.mark.asyncio
async def test_async_with_raw_response(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    client = AsyncTogether(
        api_key="fake", http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    )
    response = await client.chat.completions.with_raw_response.create(
        model="llama-3.3-70b-versatile", messages=MESSAGES
    )

    message = (await response.parse()).choices[0].message
    assert message is not None and message.content == "Paris."
    _assert_completion_attributes(in_memory_span_exporter)


def test_with_raw_response_stream_is_not_parsed(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    client = Together(
        api_key="fake", http_client=httpx.Client(transport=httpx.MockTransport(_handler))
    )
    response = client.chat.completions.with_raw_response.create(
        model="llama-3.3-70b-versatile", messages=MESSAGES, stream=True
    )

    # The stream belongs to the caller, so the span doesn't record the APIResponse as the output
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    assert spans[0].status.status_code == StatusCode.UNSET
    assert SpanAttributes.OUTPUT_VALUE not in attributes
    chunks: Any = response.parse()
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "Paris."


def test_is_parseable_without_private_options() -> None:
    # A response without the SDK's private _options must not raise into the caller's create()
    class _Response:
        is_closed = True

    assert _is_parseable(_Response(), {}) is False
