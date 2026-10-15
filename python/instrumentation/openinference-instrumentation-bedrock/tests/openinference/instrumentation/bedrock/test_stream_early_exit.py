"""Spans of streamed responses end when the stream is left before its end.

The responses are real AWS event streams served from a ``before-send`` hook, so the
botocore ``EventStream`` behaves exactly as it does against Bedrock.
"""

import base64
import gc
import json
import logging
import re
import struct
import zlib
from typing import Any, Callable, Dict, Iterator, List, Tuple

import aioboto3
import boto3
import pytest
from aioresponses import aioresponses
from botocore.awsrequest import AWSResponse
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

_CLIENT_KWARGS = {
    "region_name": "us-east-1",
    "aws_access_key_id": "123",
    "aws_secret_access_key": "321",
}


def _frame(event_type: str, payload: Dict[str, Any]) -> bytes:
    def header(name: str, value: str) -> bytes:
        n, v = name.encode(), value.encode()
        return struct.pack("B", len(n)) + n + b"\x07" + struct.pack(">H", len(v)) + v

    headers = (
        header(":event-type", event_type)
        + header(":content-type", "application/json")
        + header(":message-type", "event")
    )
    body = json.dumps(payload).encode()
    prelude = struct.pack(">II", 12 + len(headers) + len(body) + 4, len(headers))
    prelude += struct.pack(">I", zlib.crc32(prelude))
    message = prelude + headers + body
    return message + struct.pack(">I", zlib.crc32(message))


def _chunk(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    return "chunk", {"bytes": base64.b64encode(json.dumps(payload).encode()).decode()}


def _agent_chunk(text: str) -> Tuple[str, Dict[str, Any]]:
    return "chunk", {"bytes": base64.b64encode(text.encode()).decode()}


# Each stream sends "one " and then "two", so a stream left after three events has
# some of its output and a stream read to the end has all of it.
_EVENTS: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {
    "/converse-stream": [
        ("messageStart", {"role": "assistant"}),
        ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "one "}}),
        ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "two"}}),
        ("contentBlockStop", {"contentBlockIndex": 0}),
        ("messageStop", {"stopReason": "end_turn"}),
        ("metadata", {"usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5}}),
    ],
    "anthropic": [
        _chunk(
            {
                "type": "message_start",
                "message": {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "m",
                    "content": [],
                    "usage": {"input_tokens": 3, "output_tokens": 0},
                },
            }
        ),
        _chunk(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        ),
        _chunk(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "one "},
            }
        ),
        _chunk(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "two"},
            }
        ),
        _chunk({"type": "content_block_stop", "index": 0}),
        _chunk(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2},
            }
        ),
        _chunk({"type": "message_stop"}),
    ],
    "nova": [
        _chunk({"messageStart": {"role": "assistant"}}),
        _chunk({"contentBlockDelta": {"delta": {"text": "one "}, "contentBlockIndex": 0}}),
        _chunk({"contentBlockDelta": {"delta": {"text": "two"}, "contentBlockIndex": 0}}),
        _chunk({"contentBlockStop": {"contentBlockIndex": 0}}),
        _chunk({"messageStop": {"stopReason": "end_turn"}}),
        _chunk({"metadata": {"usage": {"inputTokens": 3, "outputTokens": 2}}}),
    ],
    "openai": [
        _chunk({"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}),
        _chunk({"choices": [{"index": 0, "delta": {"content": "one "}, "finish_reason": None}]}),
        _chunk({"choices": [{"index": 0, "delta": {"content": "two"}, "finish_reason": None}]}),
        _chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ],
    "/text": [_agent_chunk("one "), _agent_chunk("two"), _agent_chunk("."), _agent_chunk("")],
    "/retrieveAndGenerateStream": [
        ("output", {"text": "one "}),
        ("output", {"text": "two"}),
        ("output", {"text": "."}),
        ("output", {"text": ""}),
    ],
}

_MODEL_IDS = {
    "anthropic": "anthropic.claude-3-haiku-20240307-v1:0",
    "nova": "amazon.nova-lite-v1:0",
    "openai": "openai.gpt-oss-120b-1:0",
}


class _Raw:
    def __init__(self, frames: List[bytes]) -> None:
        self._frames = frames

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[bytes]:
        yield from self._frames

    def read(self, *args: Any, **kwargs: Any) -> bytes:
        return b"".join(self._frames)

    def close(self) -> None:
        pass

    release_conn = close


def _before_send(request: Any, **_: Any) -> AWSResponse:
    url = request.url
    if "/invoke-with-response-stream" in url:
        model_id = next(m for k, m in _MODEL_IDS.items() if m.replace(":", "%3A") in url)
        key = next(k for k, m in _MODEL_IDS.items() if m == model_id)
    else:
        key = next(k for k in _EVENTS if k.startswith("/") and url.endswith(k))
    frames = [_frame(event_type, payload) for event_type, payload in _EVENTS[key]]
    headers = {"content-type": "application/vnd.amazon.eventstream"}
    return AWSResponse(url, 200, headers, _Raw(frames))


def _client(service: str) -> Any:
    client = boto3.client(service, **_CLIENT_KWARGS)
    client.meta.events.register("before-send.*", _before_send)
    return client


def _converse_stream() -> Any:
    client = _client("bedrock-runtime")
    return client.converse_stream(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
    )["stream"]


def _invoke_model_stream(model: str) -> Callable[[], Any]:
    bodies = {
        "anthropic": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        },
        "nova": {"messages": [{"role": "user", "content": [{"text": "hi"}]}]},
        "openai": {"messages": [{"role": "user", "content": "hi"}]},
    }

    def open_stream() -> Any:
        client = _client("bedrock-runtime")
        return client.invoke_model_with_response_stream(
            modelId=_MODEL_IDS[model], body=json.dumps(bodies[model])
        )["body"]

    return open_stream


def _invoke_agent() -> Any:
    client = _client("bedrock-agent-runtime")
    return client.invoke_agent(
        agentId="AGENT", agentAliasId="ALIAS", sessionId="session", inputText="hi"
    )["completion"]


def _retrieve_and_generate_stream() -> Any:
    client = _client("bedrock-agent-runtime")
    return client.retrieve_and_generate_stream(
        input={"text": "hi"},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": "KB",
                "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/model",
            },
        },
    )["stream"]


_STREAMS = {
    "converse_stream": _converse_stream,
    "invoke_model_with_response_stream[anthropic]": _invoke_model_stream("anthropic"),
    "invoke_model_with_response_stream[nova]": _invoke_model_stream("nova"),
    "invoke_model_with_response_stream[openai]": _invoke_model_stream("openai"),
    "invoke_agent": _invoke_agent,
    "retrieve_and_generate_stream": _retrieve_and_generate_stream,
}


def _read_all(open_stream: Callable[[], Any]) -> None:
    for _ in open_stream():
        pass


def _break(open_stream: Callable[[], Any]) -> None:
    for i, _ in enumerate(open_stream()):
        if i == 2:
            break


def _close(open_stream: Callable[[], Any]) -> None:
    stream = open_stream()
    events = iter(stream)
    for _ in range(3):
        next(events)
    stream.close()


def _drop(open_stream: Callable[[], Any]) -> None:
    stream = open_stream()
    events = iter(stream)
    for _ in range(3):
        next(events)
    del events, stream


def _only_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    gc.collect()
    (span,) = exporter.get_finished_spans()
    return span


@pytest.mark.parametrize("open_stream", _STREAMS.values(), ids=_STREAMS.keys())
@pytest.mark.parametrize(
    "leave, status_code",
    [
        (_read_all, StatusCode.OK),
        (_break, StatusCode.UNSET),
        (_close, StatusCode.UNSET),
        (_drop, StatusCode.UNSET),
    ],
    ids=["read_all", "break", "close", "drop"],
)
def test_span_ends_when_stream_is_left(
    open_stream: Callable[[], Any],
    leave: Callable[[Callable[[], Any]], None],
    status_code: StatusCode,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    leave(open_stream)
    span = _only_span(in_memory_span_exporter)
    assert span.status.status_code == status_code
    output = str((span.attributes or {}).get(SpanAttributes.OUTPUT_VALUE))
    assert "one" in output
    if leave is _read_all:
        assert "two" in output


@pytest.mark.parametrize("open_stream", _STREAMS.values(), ids=_STREAMS.keys())
def test_span_does_not_end_while_stream_is_read(
    open_stream: Callable[[], Any],
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    # The stream wrapper itself is not referenced once iteration starts.
    for _ in open_stream():
        gc.collect()
        assert not in_memory_span_exporter.get_finished_spans()
    span = _only_span(in_memory_span_exporter)
    assert span.status.status_code == StatusCode.OK


def test_span_does_not_end_when_first_iterator_is_dropped(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    stream = _converse_stream()
    next(iter(stream))
    gc.collect()
    assert not in_memory_span_exporter.get_finished_spans()
    assert len(list(stream)) == len(_EVENTS["/converse-stream"]) - 1
    span = _only_span(in_memory_span_exporter)
    assert span.status.status_code == StatusCode.OK
    assert "one two" in str((span.attributes or {}).get(SpanAttributes.OUTPUT_VALUE))


def test_close_after_read_all_keeps_span(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    stream = _converse_stream()
    for _ in stream:
        pass
    stream.close()
    for _ in stream:
        pass
    del stream
    span = _only_span(in_memory_span_exporter)
    assert span.status.status_code == StatusCode.OK
    assert "one two" in str((span.attributes or {}).get(SpanAttributes.OUTPUT_VALUE))


@pytest.mark.parametrize("open_stream", _STREAMS.values(), ids=_STREAMS.keys())
def test_span_does_not_end_when_iterator_is_returned(
    open_stream: Callable[[], Any],
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    def events() -> Iterator[Any]:
        return iter(open_stream())

    # The for loop calls iter() on the returned iterator again.
    for _ in events():
        gc.collect()
        assert not in_memory_span_exporter.get_finished_spans()
    span = _only_span(in_memory_span_exporter)
    assert span.status.status_code == StatusCode.OK
    assert "two" in str((span.attributes or {}).get(SpanAttributes.OUTPUT_VALUE))


@pytest.mark.parametrize("open_stream", _STREAMS.values(), ids=_STREAMS.keys())
@pytest.mark.parametrize("close", [True, False], ids=["close", "drop"])
def test_span_ends_when_stream_is_left_before_first_event(
    open_stream: Callable[[], Any],
    close: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = open_stream()
    if close:
        stream.close()
    del stream
    span = _only_span(in_memory_span_exporter)
    assert span.status.status_code == StatusCode.UNSET
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def _async_read_all(stream: Any) -> None:
    async for _ in stream:
        pass


async def _async_break(stream: Any) -> None:
    i = 0
    async for _ in stream:
        if i == 2:
            break
        i += 1


async def _async_close(stream: Any) -> None:
    events = stream.__aiter__()
    for _ in range(3):
        await events.__anext__()
    stream.close()


async def _async_drop(stream: Any) -> None:
    events = stream.__aiter__()
    for _ in range(3):
        await events.__anext__()


@pytest.mark.parametrize(
    "leave, status_code",
    [
        (_async_read_all, StatusCode.OK),
        (_async_break, StatusCode.UNSET),
        (_async_close, StatusCode.UNSET),
        (_async_drop, StatusCode.UNSET),
    ],
    ids=["read_all", "break", "close", "drop"],
)
async def test_async_span_ends_when_stream_is_left(
    leave: Callable[[Any], Any],
    status_code: StatusCode,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    span = await _async_converse_stream_span(leave, in_memory_span_exporter)
    assert span.status.status_code == status_code
    output = str((span.attributes or {}).get(SpanAttributes.OUTPUT_VALUE))
    assert "one" in output
    if leave is _async_read_all:
        assert "two" in output


async def _async_converse_stream_span(
    leave: Callable[[Any], Any],
    exporter: InMemorySpanExporter,
) -> ReadableSpan:
    body = b"".join(_frame(t, p) for t, p in _EVENTS["/converse-stream"])
    with aioresponses() as m:
        m.post(
            re.compile(r".*/converse-stream$"),
            status=200,
            headers={"Content-Type": "application/vnd.amazon.eventstream"},
            body=body,
        )
        session = aioboto3.session.Session()
        async with session.client("bedrock-runtime", **_CLIENT_KWARGS) as client:
            response = await client.converse_stream(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                messages=[{"role": "user", "content": [{"text": "hi"}]}],
            )
            await leave(response.pop("stream"))
            del response
            return _only_span(exporter)


def _assert_closed_converse_stream_span(span: ReadableSpan) -> None:
    # Three events were read: the message start and the "one " and "two" deltas.
    assert span.name == "bedrock.converse_stream"
    assert span.status.status_code == StatusCode.UNSET
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert attributes.pop(LLM_PROVIDER) == "aws"
    assert attributes.pop(LLM_MODEL_NAME) == "anthropic.claude-3-haiku-20240307-v1:0"
    assert attributes.pop(INPUT_VALUE) == json.dumps(
        [{"role": "user", "content": [{"text": "hi"}]}]
    )
    assert attributes.pop(INPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{CONTENT_TYPE}") == "text"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{CONTENT_TEXT}") == "hi"
    assert attributes.pop(OUTPUT_VALUE) == json.dumps(
        {"role": "assistant", "content": [{"text": "one two"}]}
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{CONTENT_TYPE}") == "text"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{CONTENT_TEXT}") == "one two"
    )
    # No stop reason or usage arrived before the stream was closed.
    assert not attributes


def test_closed_converse_stream_span_attributes(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    _close(_converse_stream)
    _assert_closed_converse_stream_span(_only_span(in_memory_span_exporter))


async def test_async_closed_converse_stream_span_attributes(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    span = await _async_converse_stream_span(_async_close, in_memory_span_exporter)
    _assert_closed_converse_stream_span(span)


def test_closed_invoke_model_stream_span_attributes(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    _close(_invoke_model_stream("anthropic"))
    span = _only_span(in_memory_span_exporter)
    # Three events were read: the message start, the block start and the "one " delta.
    assert span.name == "bedrock.invoke_model_with_response_stream"
    assert span.status.status_code == StatusCode.UNSET
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert attributes.pop(LLM_MODEL_NAME) == "anthropic.claude-3-haiku-20240307-v1:0"
    assert attributes.pop(LLM_INVOCATION_PARAMETERS) == json.dumps({"max_tokens": 50})
    assert json.loads(str(attributes.pop(INPUT_VALUE))) == {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    assert attributes.pop(INPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{CONTENT_TEXT}") == "hi"
    assert json.loads(str(attributes.pop(OUTPUT_VALUE))) == {
        "id": "m",
        "type": "message",
        "role": "assistant",
        "model": "m",
        "content": [{"text": "one ", "citations": None, "type": "text"}],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 3,
            "output_tokens": 0,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None,
        },
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{CONTENT_TEXT}") == "one "
    # The message start reports the prompt tokens; the completion count comes at the end.
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 3
    assert not attributes


CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
