import json
from typing import Any, Dict, Iterator

import httpx2 as httpx
import pytest
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from openai import AsyncOpenAI, OpenAI
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.agno import AgnoInstrumentor
from openinference.semconv.trace import SpanAttributes

LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO
LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING = (
    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING
)
LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO

PROMPT, CACHE_READ, PROMPT_AUDIO = 2006, 1920, 12
COMPLETION, REASONING, COMPLETION_AUDIO = 300, 256, 24
MODEL = "o4-mini"
USAGE = {
    "prompt_tokens": PROMPT,
    "completion_tokens": COMPLETION,
    "total_tokens": PROMPT + COMPLETION,
    "prompt_tokens_details": {"cached_tokens": CACHE_READ, "audio_tokens": PROMPT_AUDIO},
    "completion_tokens_details": {
        "reasoning_tokens": REASONING,
        "audio_tokens": COMPLETION_AUDIO,
    },
}


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


def _chunk(**fields: Any) -> str:
    chunk = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": MODEL}
    return f"data: {json.dumps({**chunk, **fields})}\n\n"


def _stream_response() -> httpx.Response:
    body = "".join(
        [
            _chunk(
                choices=[
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "ok"},
                        "finish_reason": None,
                    }
                ]
            ),
            _chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]),
            _chunk(choices=[], usage=USAGE),
            "data: [DONE]\n\n",
        ]
    )
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def _completion_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "c1",
            "object": "chat.completion",
            "created": 1,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": USAGE,
        },
    )


def _handler(request: httpx.Request) -> httpx.Response:
    if json.loads(request.content).get("stream"):
        return _stream_response()
    return _completion_response()


def _llm_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = [s for s in exporter.get_finished_spans() if s.name.startswith("OpenAIChat.")]
    assert len(spans) == 1, [s.name for s in exporter.get_finished_spans()]
    return spans[0]


def _assert_counts(span: ReadableSpan) -> None:
    token_counts: Dict[str, Any] = {
        k: v for k, v in (span.attributes or {}).items() if k.startswith("llm.token_count.")
    }
    assert token_counts == {
        LLM_TOKEN_COUNT_PROMPT: PROMPT,
        LLM_TOKEN_COUNT_COMPLETION: COMPLETION,
        LLM_TOKEN_COUNT_TOTAL: PROMPT + COMPLETION,
        LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: CACHE_READ,
        LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO: PROMPT_AUDIO,
        LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING: REASONING,
        LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO: COMPLETION_AUDIO,
    }


def _sync_model() -> OpenAIChat:
    client = OpenAI(
        api_key="not-used", http_client=httpx.Client(transport=httpx.MockTransport(_handler))
    )
    return OpenAIChat(id=MODEL, client=client)


def _async_model() -> OpenAIChat:
    client = AsyncOpenAI(
        api_key="not-used",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )
    return OpenAIChat(id=MODEL, async_client=client)


def test_invoke_token_counts(in_memory_span_exporter: InMemorySpanExporter) -> None:
    Agent(model=_sync_model()).run("hi")
    _assert_counts(_llm_span(in_memory_span_exporter))


def test_invoke_stream_token_counts(in_memory_span_exporter: InMemorySpanExporter) -> None:
    for _ in Agent(model=_sync_model()).run("hi", stream=True):
        pass
    _assert_counts(_llm_span(in_memory_span_exporter))


async def test_ainvoke_token_counts(in_memory_span_exporter: InMemorySpanExporter) -> None:
    await Agent(model=_async_model()).arun("hi")
    _assert_counts(_llm_span(in_memory_span_exporter))


async def test_ainvoke_stream_token_counts(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async for _ in Agent(model=_async_model()).arun("hi", stream=True):
        pass
    _assert_counts(_llm_span(in_memory_span_exporter))
