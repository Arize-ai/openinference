"""A Responses API response that fails is an error, though its request succeeds.

The server reports the failure in the response, `{"status": "failed", "error": {...}}`, or, in a
stream, with a `response.failed` or an `error` event, after which the stream still ends normally.
Either way the span used to end with status OK. The response's status, or the reason it is
incomplete, is also recorded as its finish reason.
"""

import asyncio
import importlib
import json
from typing import Any, AsyncIterator, Dict, Iterable, Iterator, List, Mapping, Optional, cast

import pytest
from httpx import AsyncByteStream, Response
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.util.types import AttributeValue
from respx import MockRouter

from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

_BASE_URL = "https://api.openai.com/v1/"
_MODEL = "gpt-4o-mini"
_INPUT = "Tell me a long story"
_TEXT = "Once upon a"
_ERROR = {"code": "server_error", "message": "The model failed"}
_USAGE = {
    "input_tokens": 12,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 5,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 17,
}


class _MockByteStream(AsyncByteStream):
    def __init__(self, byte_stream: Iterable[bytes]) -> None:
        self._byte_stream = byte_stream

    def __iter__(self) -> Iterator[bytes]:
        yield from self._byte_stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for byte_string in self._byte_stream:
            yield byte_string


def _response(status: str, **fields: Any) -> Dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1758000000,
        "model": _MODEL,
        "status": status,
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed" if status == "completed" else "incomplete",
                "content": [{"type": "output_text", "text": _TEXT, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": _USAGE,
        **fields,
    }


def _sse(events: List[Dict[str, Any]]) -> List[bytes]:
    return [
        f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': i})}\n\n".encode()
        for i, event in enumerate(events)
    ]


def _stream(last_event: Dict[str, Any]) -> List[bytes]:
    return _sse(
        [
            {"type": "response.created", "response": {**_response("in_progress"), "output": []}},
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": _TEXT,
            },
            last_event,
        ]
    )


def _call(respx_mock: MockRouter, is_async: bool, **mock: Any) -> None:
    is_stream = "stream" in mock
    respx_mock.post(f"{_BASE_URL}responses").mock(return_value=Response(status_code=200, **mock))
    openai = importlib.import_module("openai")
    client = (openai.AsyncOpenAI if is_async else openai.OpenAI)(api_key="sk-", base_url=_BASE_URL)

    async def call_async() -> None:
        response = await client.responses.create(model=_MODEL, input=_INPUT, stream=is_stream)
        if is_stream:
            async for _ in response:
                pass

    if is_async:
        asyncio.run(call_async())
        return
    response = client.responses.create(model=_MODEL, input=_INPUT, stream=is_stream)
    if is_stream:
        for _ in response:
            pass


def _span(exporter: InMemorySpanExporter) -> ReadableSpan:
    (span,) = [
        span
        for span in exporter.get_finished_spans()
        if span.instrumentation_scope is not None
        and span.instrumentation_scope.name == "openinference.instrumentation.openai"
    ]
    return span


def _pop_request_attributes(attributes: Dict[str, Any], is_stream: bool) -> None:
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.LLM.value
    )
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "openai"
    assert attributes.pop(SpanAttributes.LLM_SYSTEM) == "openai"
    assert json.loads(attributes.pop(SpanAttributes.INPUT_VALUE)) == {
        "model": _MODEL,
        "input": _INPUT,
        "stream": is_stream,
    }
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == OpenInferenceMimeTypeValues.JSON.value
    assert json.loads(attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS)) == {
        "model": _MODEL,
        "stream": is_stream,
    }
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == _MODEL
    assert attributes.pop(f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.message.role") == "user"
    assert attributes.pop(f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.message.content") == _INPUT


def _pop_response_attributes(attributes: Dict[str, Any], has_output: bool) -> None:
    assert isinstance(attributes.pop(SpanAttributes.OUTPUT_VALUE), str)
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == OpenInferenceMimeTypeValues.JSON.value
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 12
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 5
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 17
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING) == 0
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == 0
    if has_output:
        message = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0."
        content = f"{message}{MessageAttributes.MESSAGE_CONTENTS}.0."
        assert attributes.pop(f"{message}{MessageAttributes.MESSAGE_ROLE}") == "assistant"
        assert attributes.pop(f"{content}{MessageContentAttributes.MESSAGE_CONTENT_TYPE}") == "text"
        assert attributes.pop(f"{content}{MessageContentAttributes.MESSAGE_CONTENT_TEXT}") == _TEXT


def _attributes(span: ReadableSpan) -> Dict[str, Any]:
    return dict(cast(Mapping[str, AttributeValue], span.attributes))


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "response, status_code, description, finish_reason",
    [
        pytest.param(_response("completed"), StatusCode.OK, None, "stop", id="completed"),
        pytest.param(
            _response("incomplete", incomplete_details={"reason": "max_output_tokens"}),
            StatusCode.OK,
            None,
            "length",
            id="incomplete",
        ),
        pytest.param(
            {**_response("failed", error=_ERROR), "output": []},
            StatusCode.ERROR,
            "The model failed",
            "error",
            id="failed",
        ),
    ],
)
def test_a_response_records_its_status_and_finish_reason(
    response: Dict[str, Any],
    status_code: StatusCode,
    description: Optional[str],
    finish_reason: str,
    is_async: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    _call(respx_mock, is_async, json=response)

    span = _span(in_memory_span_exporter)
    assert span.status.status_code == status_code
    assert span.status.description == description
    attributes = _attributes(span)
    _pop_request_attributes(attributes, is_stream=False)
    _pop_response_attributes(attributes, has_output=bool(response["output"]))
    assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == finish_reason
    assert attributes == {}


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "last_event",
    [
        pytest.param(
            {"type": "response.failed", "response": {**_response("failed"), "error": _ERROR}},
            id="response.failed",
        ),
        pytest.param({"type": "error", **_ERROR, "param": None}, id="error"),
    ],
)
def test_a_stream_that_reports_a_failure_is_an_error(
    last_event: Dict[str, Any],
    is_async: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    _call(respx_mock, is_async, stream=_MockByteStream(_stream(last_event)))

    span = _span(in_memory_span_exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "The model failed"


@pytest.mark.parametrize("is_async", [False, True])
def test_a_stream_that_completes_is_ok_and_records_its_finish_reason(
    is_async: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    last_event = {"type": "response.completed", "response": _response("completed")}

    _call(respx_mock, is_async, stream=_MockByteStream(_stream(last_event)))

    span = _span(in_memory_span_exporter)
    assert span.status.status_code == StatusCode.OK
    assert not span.status.description
    attributes = _attributes(span)
    _pop_request_attributes(attributes, is_stream=True)
    _pop_response_attributes(attributes, has_output=True)
    assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "stop"
    assert attributes == {}
