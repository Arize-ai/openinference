"""Tests for coverage of the current Messages API: robust extraction of input messages, and
streams whose events fail to accumulate."""

import json
from typing import Any, Dict, List

import httpx2
import pytest
from anthropic import Anthropic
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.anthropic._wrappers import _get_llm_input_messages
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolCallAttributes,
)

LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON

MODEL = "claude-sonnet-4-6"


def _mock_anthropic_client(handler: Any) -> Anthropic:
    """Build an ``Anthropic`` client whose HTTP transport is mocked by ``handler``."""
    transport = httpx2.MockTransport(handler)
    return Anthropic(api_key="sk-ant-fake", http_client=httpx2.Client(transport=transport))


def _message_json(content: List[Dict[str, Any]], **fields: Any) -> Dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": content,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 1},
        **fields,
    }


def _event_stream_response(events: List[Dict[str, Any]]) -> Any:
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    return httpx2.Response(
        status_code=200, headers={"content-type": "text/event-stream"}, content=body.encode()
    )


def _get_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    (span,) = exporter.get_finished_spans()
    return span


def test_input_messages_survive_a_bad_message() -> None:
    attributes = dict(
        _get_llm_input_messages(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant"},  # no content, so extracting it fails
                {"role": "user", "content": "third"},
            ],
            system="be brief",
        )
    )
    assert attributes == {
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}": "system",
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}": "be brief",
        f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}": "user",
        f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENT}": "first",
        f"{LLM_INPUT_MESSAGES}.3.{MESSAGE_ROLE}": "user",
        f"{LLM_INPUT_MESSAGES}.3.{MESSAGE_CONTENT}": "third",
    }


def test_input_messages_survive_a_bad_content_block() -> None:
    attributes = dict(
        _get_llm_input_messages(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text"},  # no text, so extracting it fails
                        {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"a": 1}},
                        {"type": "text", "text": "after"},
                    ],
                },
            ],
            system=[{"type": "text"}, {"type": "text", "text": "be brief"}],
        )
    )
    assert attributes == {
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}": "system",
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}": "text",
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TEXT}": "be brief",
        f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}": "assistant",
        f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_ID}": "toolu_1",
        f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}": "f",
        f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}": (
            '{"a": 1}'
        ),
        f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}": "tool_use",
        f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.1.{TOOL_CALL_ID}": "toolu_1",
        f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.1.{TOOL_CALL_FUNCTION_NAME}": "f",
        f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.1.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}": (
            '{"a": 1}'
        ),
        f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.2.{MESSAGE_CONTENT_TYPE}": "text",
        f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.2.{MESSAGE_CONTENT_TEXT}": "after",
    }


def _tool_use_stream_handler(request: Any) -> Any:
    return _event_stream_response(
        [
            {"type": "message_start", "message": _message_json([])},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}},
            },
            *(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": partial_json},
                }
                for partial_json in ['{"city": ', '"Par', 'is", "days": 3}']
            ),
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 9},
            },
            {"type": "message_stop"},
        ]
    )


@pytest.mark.parametrize("failing_delta", [1, 2, 3])
def test_failed_stream_event_does_not_corrupt_tool_call_arguments(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
    monkeypatch: pytest.MonkeyPatch,
    failing_delta: int,
) -> None:
    from anthropic.lib.streaming import _messages

    original_accumulate_event = _messages.accumulate_event
    deltas = 0

    def accumulate_event(**kwargs: Any) -> Any:
        # fails like an unparseable partial JSON would, after the SDK has read the delta but
        # before it has buffered it
        nonlocal deltas
        if kwargs["event"].type == "content_block_delta":
            deltas += 1
            if deltas == failing_delta:
                raise ValueError("Unable to parse tool parameter JSON from model.")
        return original_accumulate_event(**kwargs)

    monkeypatch.setattr(_messages, "accumulate_event", accumulate_event)
    client = _mock_anthropic_client(_tool_use_stream_handler)
    stream = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": "weather?"}],
        stream=True,
    )
    for _ in stream:
        pass

    attributes = dict(_get_span(in_memory_span_exporter).attributes or {})
    tool_call = f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0"
    assert json.loads(str(attributes[f"{tool_call}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"])) == {
        "city": "Paris",
        "days": 3,
    }
    assert attributes[SpanAttributes.LLM_FINISH_REASON] == "tool_use"
