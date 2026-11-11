"""The LLM span of a streaming chat completion must cover the stream, not the call.

``create(stream=True)`` returns before any content arrives. Recording the stream
object as the output and ending the span at that point leaves every streaming
call with a zero-length span whose output is a Python repr, no model name, no
token counts and no finish reason.
"""

import asyncio
import gc
import json
from typing import Any, AsyncIterator, Dict, List

import httpx
import pytest
from groq import AsyncGroq, Groq
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from openinference.instrumentation import (
    REDACTED_VALUE,
    TraceConfig,
    suppress_tracing,
    using_attributes,
)
from openinference.instrumentation.groq import GroqInstrumentor
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolCallAttributes,
)

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


def _sse_body(usage_location: str = "usage") -> bytes:
    usage_chunk = _chunk(choices=[], usage=None if usage_location == "x_groq" else USAGE)
    if usage_location != "usage":
        usage_chunk["x_groq"] = {
            "usage": (
                {**USAGE, "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
                if usage_location == "both"
                else USAGE
            )
        }
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
        usage_chunk,
        "data: [DONE]\n\n",
    ]
    body = "".join(
        event if isinstance(event, str) else f"data: {json.dumps(event)}\n\n" for event in events
    )
    return body.encode()


def _handler(request: httpx.Request, usage_location: str = "usage") -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=_sse_body(usage_location),
    )


def _client(usage_location: str = "usage") -> Groq:
    return Groq(
        api_key="fake",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: _handler(request, usage_location))
        ),
    )


def _async_client(usage_location: str = "usage") -> AsyncGroq:
    return AsyncGroq(
        api_key="fake",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: _handler(request, usage_location))
        ),
    )


def _chat_attributes(spans: List[Any]) -> Dict[str, Any]:
    (span,) = spans
    return dict(span.attributes or {})


def _client_from_chunks(chunks: List[Dict[str, Any]]) -> Groq:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    body += "data: [DONE]\n\n"
    return Groq(
        api_key="fake",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=body.encode(),
                )
            )
        ),
    )


def _raw_chunk(
    delta: Dict[str, Any],
    *,
    finish_reason: Any = None,
    usage: Any = None,
    choices: Any = None,
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
                    "delta": delta,
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


@pytest.mark.usefixtures("setup_groq_instrumentation")
@pytest.mark.parametrize("usage_location", ["usage", "x_groq", "both"])
def test_stream_span_ends_after_consumption(
    in_memory_span_exporter: InMemorySpanExporter,
    usage_location: str,
) -> None:
    stream = _client(usage_location).chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from one to three."}],
        stream=True,
        extra_body=(
            {"stream_options": {"include_usage": True}} if usage_location != "x_groq" else None
        ),
    )
    # Nothing has been read yet, so the span must not be finished.
    assert in_memory_span_exporter.get_finished_spans() == ()

    chunks = list(stream)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.OK
    attributes = dict(span.attributes or {})
    output_value = attributes[SpanAttributes.OUTPUT_VALUE]
    assert isinstance(output_value, str)
    assert json.loads(output_value)["choices"][0]["message"]["content"] == "1, 2"
    assert attributes[SpanAttributes.LLM_MODEL_NAME] == MODEL
    assert attributes[SpanAttributes.LLM_FINISH_REASON] == "stop"
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == USAGE["total_tokens"]
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == USAGE["prompt_tokens"]
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == USAGE["completion_tokens"]
    assert attributes[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{'message.role'}"] == "assistant"


@pytest.mark.usefixtures("setup_groq_instrumentation")
def test_stream_keeps_reasoning_and_usage_details(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    chunks = [
        _raw_chunk({"role": "assistant", "content": "A", "reasoning": "R"}),
        _raw_chunk({}, finish_reason="stop"),
        _raw_chunk(
            {},
            choices=[],
            usage={
                **USAGE,
                "prompt_tokens_details": {"cached_tokens": 5},
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        ),
    ]
    stream = _client_from_chunks(chunks).chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    list(stream)

    attributes = _chat_attributes(list(in_memory_span_exporter.get_finished_spans()))
    output = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    contents = f"{output}.{MessageAttributes.MESSAGE_CONTENTS}"
    assert (
        attributes[f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"] == "reasoning"
    )
    assert attributes[f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"] == "R"
    assert attributes[f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"] == "text"
    assert attributes[f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"] == "A"
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] == 5
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING] == 7


@pytest.mark.usefixtures("setup_groq_instrumentation")
def test_stream_accumulates_tool_call_arguments(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    chunks = [
        _raw_chunk(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"x"'},
                    }
                ],
            }
        ),
        _raw_chunk(
            {
                "tool_calls": [
                    {"index": 0, "function": {"arguments": ": 1}"}},
                ]
            },
            finish_reason="tool_calls",
        ),
    ]
    stream = _client_from_chunks(chunks).chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    list(stream)

    attributes = _chat_attributes(list(in_memory_span_exporter.get_finished_spans()))
    tool_call = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
        f".{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
    )
    arguments = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
        f".{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    )
    assert attributes[tool_call] == "add"
    assert attributes[arguments] == '{"x": 1}'


@pytest.mark.usefixtures("setup_groq_instrumentation")
def test_stream_respects_suppress_tracing(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with suppress_tracing():
        stream = _client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Count from one to three."}],
            stream=True,
        )
        chunks = list(stream)

    assert (
        "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "1, 2"
    )
    assert in_memory_span_exporter.get_finished_spans() == ()


@pytest.mark.usefixtures("setup_groq_instrumentation")
def test_stream_retains_context_attributes(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with using_attributes(session_id="stream-session", user_id="stream-user"):
        stream = _client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Count from one to three."}],
            stream=True,
        )
    # Consume after leaving the context: deferred span finalization must retain it.
    list(stream)

    (span,) = in_memory_span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes[SpanAttributes.SESSION_ID] == "stream-session"
    assert attributes[SpanAttributes.USER_ID] == "stream-user"
    output_value = attributes[SpanAttributes.OUTPUT_VALUE]
    assert isinstance(output_value, str)
    assert json.loads(output_value)["choices"][0]["message"]["content"] == "1, 2"
    assert (
        attributes[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == "1, 2"
    )


@pytest.mark.parametrize("hide_outputs", [False, True])
def test_stream_output_respects_trace_config(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    hide_outputs: bool,
) -> None:
    instrumentor = GroqInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_outputs=hide_outputs),
    )
    try:
        stream = _client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Count from one to three."}],
            stream=True,
        )
        chunks = list(stream)
    finally:
        instrumentor.uninstrument()

    assert (
        "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "1, 2"
    )
    (span,) = in_memory_span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    if hide_outputs:
        assert attributes[SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE
        assert SpanAttributes.OUTPUT_MIME_TYPE not in attributes
        assert not any(
            key.startswith(f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.") for key in attributes
        )
    else:
        output_value = attributes[SpanAttributes.OUTPUT_VALUE]
        assert isinstance(output_value, str)
        assert json.loads(output_value)["choices"][0]["message"]["content"] == "1, 2"
        assert (
            attributes[
                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"
            ]
            == "1, 2"
        )
    assert attributes[SpanAttributes.LLM_MODEL_NAME] == MODEL
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == USAGE["total_tokens"]


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
    output_value = attributes[SpanAttributes.OUTPUT_VALUE]
    assert isinstance(output_value, str)
    assert json.loads(output_value)["choices"][0]["message"]["content"] == "1"


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
    output_value = attributes[SpanAttributes.OUTPUT_VALUE]
    assert isinstance(output_value, str)
    assert json.loads(output_value)["choices"][0]["message"]["content"] == "1"


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
@pytest.mark.parametrize("usage_location", ["usage", "x_groq", "both"])
async def test_async_stream_span_ends_after_consumption(
    in_memory_span_exporter: InMemorySpanExporter,
    usage_location: str,
) -> None:
    stream = await _async_client(usage_location).chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from one to three."}],
        stream=True,
        extra_body=(
            {"stream_options": {"include_usage": True}} if usage_location != "x_groq" else None
        ),
    )
    assert in_memory_span_exporter.get_finished_spans() == ()

    chunks = [chunk async for chunk in stream]
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.OK
    attributes = dict(span.attributes or {})
    output_value = attributes[SpanAttributes.OUTPUT_VALUE]
    assert isinstance(output_value, str)
    assert json.loads(output_value)["choices"][0]["message"]["content"] == "1, 2"
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == USAGE["total_tokens"]


@pytest.mark.usefixtures("setup_groq_instrumentation")
@pytest.mark.asyncio
async def test_async_stream_cancelled_during_read_ends_the_span(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    read_started = asyncio.Event()

    class BlockingStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield f"data: {json.dumps(_chunk('1'))}\n\n".encode()
            read_started.set()
            await asyncio.Event().wait()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=BlockingStream(),
        )

    async with AsyncGroq(
        api_key="fake",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Count from one to three."}],
            stream=True,
        )
        try:
            assert (await stream.__anext__()).choices[0].delta.content == "1"
            task = asyncio.create_task(stream.__anext__())
            await asyncio.wait_for(read_started.wait(), timeout=1)
            task.cancel("cancelled by caller")
            with pytest.raises(asyncio.CancelledError):
                await task

            # Keep the stream alive: cancellation must finish the span now,
            # without depending on close() or garbage collection.
            (span,) = in_memory_span_exporter.get_finished_spans()
            assert span.status.status_code == StatusCode.ERROR
            assert span.status.description == "CancelledError: cancelled by caller"
            (exception,) = (event for event in span.events if event.name == "exception")
            assert exception.attributes is not None
            assert str(exception.attributes["exception.type"]).endswith("CancelledError")
            assert exception.attributes["exception.message"] == "cancelled by caller"
            attributes = dict(span.attributes or {})
            output_value = attributes[SpanAttributes.OUTPUT_VALUE]
            assert isinstance(output_value, str)
            assert json.loads(output_value)["choices"][0]["message"]["content"] == "1"
        finally:
            await stream.close()
        assert len(in_memory_span_exporter.get_finished_spans()) == 1


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
    output_value = attributes[SpanAttributes.OUTPUT_VALUE]
    assert isinstance(output_value, str)
    assert json.loads(output_value)["choices"][0]["message"]["content"] == "1"
