"""Anthropic's ``usage.input_tokens`` excludes cache reads and writes, so
``llm.token_count.prompt`` must fold them back in (spec/semantic_conventions.md)."""

import json
from typing import Any, Dict, Iterator

import httpx2 as httpx
import pytest
from agno.agent import Agent
from agno.models.anthropic import Claude
from anthropic import Anthropic, AsyncAnthropic
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.agno import AgnoInstrumentor
from openinference.semconv.trace import SpanAttributes

LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE = (
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE
)

INPUT, CACHE_READ, CACHE_WRITE, OUTPUT = 4, 13210, 38, 29
PROMPT = INPUT + CACHE_READ + CACHE_WRITE
MODEL = "claude-sonnet-5-5"


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(autouse=True)
def instrument(in_memory_span_exporter: InMemorySpanExporter) -> Iterator[None]:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    AgnoInstrumentor().instrument(tracer_provider=provider)
    yield
    AgnoInstrumentor().uninstrument()


def _usage(output_tokens: int = OUTPUT) -> Dict[str, Any]:
    return {
        "input_tokens": INPUT,
        "cache_read_input_tokens": CACHE_READ,
        "cache_creation_input_tokens": CACHE_WRITE,
        "output_tokens": output_tokens,
    }


def _message_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": _usage(),
        },
    )


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps({'type': event, **data})}\n\n"


def _stream_response() -> httpx.Response:
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": _usage(output_tokens=1),
    }
    body = "".join(
        [
            _sse("message_start", {"message": message}),
            _sse(
                "content_block_start",
                {"index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            _sse(
                "content_block_delta",
                {"index": 0, "delta": {"type": "text_delta", "text": "ok"}},
            ),
            _sse("content_block_stop", {"index": 0}),
            _sse(
                "message_delta",
                {
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": _usage(),
                },
            ),
            _sse("message_stop", {}),
        ]
    )
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def _handler(request: httpx.Request) -> httpx.Response:
    if json.loads(request.content).get("stream"):
        return _stream_response()
    return _message_response()


def _llm_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = [s for s in exporter.get_finished_spans() if s.name.startswith("Claude.")]
    assert len(spans) == 1, [s.name for s in exporter.get_finished_spans()]
    return spans[0]


def _token_counts(span: ReadableSpan) -> Dict[str, Any]:
    return {k: v for k, v in (span.attributes or {}).items() if k.startswith("llm.token_count.")}


def _assert_counts(span: ReadableSpan, *, total: bool) -> None:
    expected: Dict[str, Any] = {
        LLM_TOKEN_COUNT_PROMPT: PROMPT,
        LLM_TOKEN_COUNT_COMPLETION: OUTPUT,
        LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: CACHE_READ,
        LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: CACHE_WRITE,
    }
    if total:
        expected[LLM_TOKEN_COUNT_TOTAL] = PROMPT + OUTPUT
    assert _token_counts(span) == expected


def _sync_model() -> Claude:
    client = Anthropic(
        api_key="not-used", http_client=httpx.Client(transport=httpx.MockTransport(_handler))
    )
    return Claude(id=MODEL, client=client)


def _async_model() -> Claude:
    client = AsyncAnthropic(
        api_key="not-used",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )
    return Claude(id=MODEL, async_client=client)


def test_invoke_prompt_tokens_include_cache_tokens(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    Agent(model=_sync_model()).run("hi")
    _assert_counts(_llm_span(in_memory_span_exporter), total=False)


def test_invoke_stream_prompt_tokens_include_cache_tokens(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    for _ in Agent(model=_sync_model()).run("hi", stream=True):
        pass
    _assert_counts(_llm_span(in_memory_span_exporter), total=False)


async def test_ainvoke_prompt_and_total_tokens_include_cache_tokens(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    await Agent(model=_async_model()).arun("hi")
    _assert_counts(_llm_span(in_memory_span_exporter), total=True)


async def test_ainvoke_stream_prompt_tokens_include_cache_tokens(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async for _ in Agent(model=_async_model()).arun("hi", stream=True):
        pass
    _assert_counts(_llm_span(in_memory_span_exporter), total=False)
