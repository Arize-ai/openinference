"""A content block that the SDK fails to accumulate, as it does for a block type it does not know
under Pydantic v1, must not drop the block nor shift the blocks after it, whose deltas refer to
them by index."""

import json
from typing import Any, Dict, List

import httpx2
import pytest
from anthropic import Anthropic
from anthropic.lib.streaming import _beta_messages, _messages
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)

LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT

_EVENTS: List[Dict[str, Any]] = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 1},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "brand_new_block"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "text_delta", "text": "after"},
    },
    {"type": "content_block_stop", "index": 1},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 2},
    },
    {"type": "message_stop"},
]


def _handler(request: Any) -> Any:
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in _EVENTS)
    return httpx2.Response(
        status_code=200, headers={"content-type": "text/event-stream"}, content=body.encode()
    )


@pytest.mark.parametrize("is_beta", [False, True])
def test_block_the_sdk_fails_to_accumulate_is_kept_at_its_index(
    is_beta: bool,
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    module: Any = _beta_messages if is_beta else _messages
    accumulate_event = module.accumulate_event

    def failing_accumulate_event(*, event: Any, **kwargs: Any) -> Any:
        if event.type == "content_block_start" and event.content_block.type == "brand_new_block":
            # as the SDK raises for a block type it does not know under Pydantic v1
            raise AttributeError("__fields__")
        return accumulate_event(event=event, **kwargs)

    monkeypatch.setattr(module, "accumulate_event", failing_accumulate_event)
    client = Anthropic(
        api_key="sk-ant-fake", http_client=httpx2.Client(transport=httpx2.MockTransport(_handler))
    )
    messages: Any = client.beta.messages if is_beta else client.messages
    for _ in messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    ):
        pass
    (span,) = in_memory_span_exporter.get_finished_spans()
    attributes: Dict[str, Any] = dict(span.attributes or {})
    contents = f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}"
    assert attributes[f"{contents}.0.{MESSAGE_CONTENT_TYPE}"] == "brand_new_block"
    assert json.loads(attributes[f"{contents}.0.{MESSAGE_CONTENT_TEXT}"]) == {
        "type": "brand_new_block"
    }
    assert attributes[f"{contents}.1.{MESSAGE_CONTENT_TYPE}"] == "text"
    assert attributes[f"{contents}.1.{MESSAGE_CONTENT_TEXT}"] == "after"
