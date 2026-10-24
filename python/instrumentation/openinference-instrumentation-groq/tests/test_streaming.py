"""The LLM span of a streaming chat completion must cover the stream, not the call.

``create(stream=True)`` returns before any content arrives. Recording the stream
object as the output and ending the span at that point leaves every streaming
call with a zero-length span whose output is a Python repr, no model name, no
token counts and no finish reason.
"""

import gc
import json
from typing import Any, Dict, List

import httpx
import pytest
from groq import AsyncGroq, Groq
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from openinference.semconv.trace import SpanAttributes

MODEL = "fake_model"

USAGE = {
    "prompt_tokens": 25,
    "completion_tokens": 379,
    "total_tokens": 404,
    "prompt_time": 0.0025,
    "completion_time": 0.616,
    "total_time": 0.619,
    "queue_time": None,
}


def _chunk(
    content: str = "",
    *,
    finish_reason: Any = None,
    choices: Any = None,
    usage: Any = None,
) -> Dict[str, Any]:
    return {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "created": 1722531851,
        "model": MODEL,
        "system_fingerprint": "fp0",
        "choices": (
            [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ]
            if choices is None
            else choices
        ),
        "usage": usage,
        "x_groq": None,
    }


def _sse_body() -> bytes:
    events: List[Any] = [
        _chunk("1"),
        _chunk(", "),
        _chunk("2"),
        _chunk(
            choices=[
                {
                    "index": 0,
                    "delta": {},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ]
        ),
        _chunk(choices=[], usage=USAGE),
        "data: [DONE]\n\n",
    ]
    body = "".join(
        event if isinstance(event, str) else f"data: {json.dumps(event)}\n\n" for event in events
    )
    return body.encode()


def _handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=_sse_body(),
    )


def _client() -> Groq:
    return Groq(api_key="fake", http_client=httpx.Client(transport=httpx.MockTransport(_handler)))


def _async_client() -> AsyncGroq:
    return AsyncGroq(
        api_key="fake",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )


def _chat_attributes(spans: List[Any]) -> Dict[str, Any]:
    (span,) = spans
    return dict(span.attributes or {})


@pytest.mark.usefixtures("setup_groq_instrumentation")
def test_stream_span_ends_after_consumption(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    stream = _client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from one to three."}],
        stream=True,
    )
    # Nothing has been read yet, so the span must not be finished.
    assert in_memory_span_exporter.get_finished_spans() == ()

    chunks = list(stream)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.OK
    attributes = dict(span.attributes or {})
    assert attributes[SpanAttributes.OUTPUT_VALUE]
    assert json.loads(attributes[SpanAttributes.OUTPUT_VALUE])["choices"][0]["message"][
        "content"
    ] == ("1, 2")
    assert attributes[SpanAttributes.LLM_MODEL_NAME] == MODEL
    assert attributes[SpanAttributes.LLM_FINISH_REASON] == "stop"
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == USAGE["total_tokens"]
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == USAGE["prompt_tokens"]
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == USAGE["completion_tokens"]
    assert attributes[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{'message.role'}"] == "assistant"


@pytest.mark.usefixtures("setup_groq_instrumentation")
def test_stream_closed_early_ends_the_span_once(
    in_memory_span_exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from one to three."}],
        stream=True,
    )
    for _ in stream:
        break
    stream.close()
    del stream
    gc.collect()

    assert "Calling end() on an ended span." not in caplog.text
    (span,) = in_memory_span_exporter.get_finished_spans()
    # Closed before exhaustion: UNSET tells a truncated stream apart from a
    # fully consumed one.
    assert span.status.status_code == StatusCode.UNSET
    attributes = dict(span.attributes or {})
    assert (
        json.loads(attributes[SpanAttributes.OUTPUT_VALUE])["choices"][0]["message"]["content"]
        == "1"
    )


@pytest.mark.usefixtures("setup_groq_instrumentation")
def test_stream_abandoned_after_partial_iteration_ends_the_span(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    stream = _client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from one to three."}],
        stream=True,
    )
    for _ in stream:
        break
    del stream
    gc.collect()

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.UNSET
    attributes = dict(span.attributes or {})
    assert (
        json.loads(attributes[SpanAttributes.OUTPUT_VALUE])["choices"][0]["message"]["content"]
        == "1"
    )


@pytest.mark.usefixtures("setup_groq_instrumentation")
def test_stream_context_manager_iterates_through_the_proxy(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    stream = _client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from one to three."}],
        stream=True,
    )
    with stream as entered:
        # The SDK's Stream returns itself from __enter__; iteration inside the
        # block must still go through the traced proxy.
        assert entered is stream
        assert in_memory_span_exporter.get_finished_spans() == ()
        chunks = list(entered)
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.OK
    attributes = dict(span.attributes or {})
    assert attributes[SpanAttributes.LLM_FINISH_REASON] == "stop"


@pytest.mark.usefixtures("setup_groq_instrumentation")
@pytest.mark.asyncio
async def test_async_stream_span_ends_after_consumption(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    stream = await _async_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from one to three."}],
        stream=True,
    )
    assert in_memory_span_exporter.get_finished_spans() == ()

    chunks = [chunk async for chunk in stream]
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.OK
    attributes = dict(span.attributes or {})
    assert (
        json.loads(attributes[SpanAttributes.OUTPUT_VALUE])["choices"][0]["message"]["content"]
        == "1, 2"
    )
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == USAGE["total_tokens"]


@pytest.mark.usefixtures("setup_groq_instrumentation")
@pytest.mark.asyncio
async def test_async_stream_closed_early_ends_the_span(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    stream = await _async_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from one to three."}],
        stream=True,
    )
    async for _ in stream:
        break
    await stream.close()

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.UNSET
    attributes = dict(span.attributes or {})
    assert (
        json.loads(attributes[SpanAttributes.OUTPUT_VALUE])["choices"][0]["message"]["content"]
        == "1"
    )
