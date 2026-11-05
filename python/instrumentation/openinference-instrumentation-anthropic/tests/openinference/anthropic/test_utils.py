import json
import logging
from typing import Any, Dict, Iterator, List, Tuple

import httpx2
import pytest
from anthropic import Anthropic
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.anthropic._utils import _finish_tracing
from openinference.instrumentation.anthropic._with_span import _WithSpan
from openinference.semconv.trace import SpanAttributes

LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ


class _Attributes:
    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        yield "llm.system", "anthropic"


def _failing_params() -> Iterator[Tuple[str, AttributeValue]]:
    yield "llm.provider", "anthropic"
    raise RuntimeError("span finalization blew up")


def _message(usage: Dict[str, Any], *, streaming: bool) -> Dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [] if streaming else [{"type": "text", "text": "Paris"}],
        "stop_reason": None if streaming else "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }


def _sse(usage: Dict[str, Any]) -> bytes:
    """A stream whose message_start carries `usage` and whose message_delta reports the output."""
    start_usage = {**usage, "output_tokens": 1}
    events: List[Dict[str, Any]] = [
        {"type": "message_start", "message": _message(start_usage, streaming=True)},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Paris"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": usage["output_tokens"]},
        },
        {"type": "message_stop"},
    ]
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


@pytest.mark.parametrize("stream", [False, True], ids=["create", "stream"])
@pytest.mark.parametrize(
    "usage,expected",
    [
        pytest.param(
            {"input_tokens": None, "output_tokens": 5},
            {LLM_TOKEN_COUNT_COMPLETION: 5},
            id="input-null",
        ),
        pytest.param(
            {"input_tokens": None, "output_tokens": 5, "cache_read_input_tokens": 100},
            {LLM_TOKEN_COUNT_COMPLETION: 5, LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 100},
            id="input-null-with-cache-read",
        ),
        pytest.param(
            {"input_tokens": 12, "output_tokens": None},
            {LLM_TOKEN_COUNT_PROMPT: 12},
            id="output-null",
        ),
        pytest.param({"input_tokens": None, "output_tokens": None}, {}, id="all-null"),
    ],
)
def test_null_usage_counts_are_left_out(
    stream: bool,
    usage: Dict[str, Any],
    expected: Dict[str, int],
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The SDK does not validate responses, so a proxy can return a null count. A null count is
    unknown: it must not be recorded as 0, nor summed into a total, and the counts that were
    reported must still be recorded."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if stream:
            return httpx2.Response(
                200, content=_sse(usage), headers={"content-type": "text/event-stream"}
            )
        return httpx2.Response(200, json=_message(usage, streaming=False))

    client = Anthropic(
        api_key="sk-ant-fake", http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    with caplog.at_level(logging.ERROR):
        if stream:
            for _ in client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=16,
                messages=[{"role": "user", "content": "Capital of France?"}],
                stream=True,
            ):
                pass
        else:
            client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=16,
                messages=[{"role": "user", "content": "Capital of France?"}],
            )

    (span,) = in_memory_span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    token_counts = {k: v for k, v in attributes.items() if k.startswith("llm.token_count.")}
    assert token_counts == expected
    assert attributes[SpanAttributes.LLM_MODEL_NAME] == "claude-sonnet-4-6"
    assert "Failed to get" not in caplog.text


def test_finish_tracing_logs_and_recovers_when_finalization_fails(
    tracer_provider: TracerProvider,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An instrumentation failure while ending the span must not reach user code."""
    span = tracer_provider.get_tracer(__name__).start_span("Test")
    with_span = _WithSpan(span=span, params=_failing_params())

    with caplog.at_level(logging.ERROR):
        _finish_tracing(with_span=with_span, has_attributes=_Attributes())

    assert "Failed to finish tracing" in caplog.text
