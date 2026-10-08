import base64
from typing import Any, Dict, cast

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.bedrock._converse_attributes import (
    get_attributes_from_response_data,
    get_message_objects,
)
from openinference.instrumentation.bedrock._converse_stream_callback import (
    _ConverseStreamCallback,
)
from openinference.semconv.trace import MessageContentAttributes

MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_CONTENT_SIGNATURE = MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE
MESSAGE_CONTENT_DATA = MessageContentAttributes.MESSAGE_CONTENT_DATA
MESSAGE_CONTENT_ID = MessageContentAttributes.MESSAGE_CONTENT_ID


def test_get_message_objects_visible_reasoning_preserves_order() -> None:
    message_list = [
        {
            "role": "assistant",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": "Let me think about this...",
                            "signature": "sig-123",
                        }
                    }
                },
                {"text": "The answer is 42."},
            ],
        }
    ]
    messages = get_message_objects(message_list)  # type: ignore[arg-type]
    assert len(messages) == 1
    contents = messages[0]["contents"]
    assert len(contents) == 2
    assert contents[0] == {
        "type": "reasoning",
        "text": "Let me think about this...",
        "signature": "sig-123",
    }
    assert contents[1] == {"type": "text", "text": "The answer is 42."}


def test_get_message_objects_redacted_reasoning() -> None:
    redacted_bytes = b"\x00\x01encrypted-payload"
    message_list = [
        {
            "role": "assistant",
            "content": [
                {"reasoningContent": {"redactedContent": redacted_bytes}},
                {"text": "Final answer."},
            ],
        }
    ]
    messages = get_message_objects(message_list)  # type: ignore[arg-type]
    contents = messages[0]["contents"]
    assert contents[0] == {
        "type": "reasoning",
        "data": base64.b64encode(redacted_bytes).decode("utf-8"),
    }
    assert contents[1] == {"type": "text", "text": "Final answer."}


def test_get_attributes_from_response_data_emits_reasoning_attributes_no_id() -> None:
    request_data: Dict[str, Any] = {"modelId": "anthropic.claude-3", "messages": []}
    response_data: Dict[str, Any] = {
        "stopReason": "end_turn",
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {"text": "Reasoning...", "signature": "sig-abc"}
                        }
                    },
                    {"text": "Answer."},
                ],
            }
        },
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    }
    attributes = get_attributes_from_response_data(
        cast(Any, request_data), cast(Any, response_data)
    )
    assert attributes[f"llm.output_messages.0.message.contents.0.{MESSAGE_CONTENT_TYPE}"] == (
        "reasoning"
    )
    assert (
        attributes[f"llm.output_messages.0.message.contents.0.{MESSAGE_CONTENT_TEXT}"]
        == "Reasoning..."
    )
    assert (
        attributes[f"llm.output_messages.0.message.contents.0.{MESSAGE_CONTENT_SIGNATURE}"]
        == "sig-abc"
    )
    assert f"llm.output_messages.0.message.contents.0.{MESSAGE_CONTENT_ID}" not in attributes


def test_converse_stream_callback_accumulates_reasoning_content(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    tracer = tracer_provider.get_tracer(__name__)
    span = tracer.start_span("converse.stream.reasoning")
    request: Dict[str, Any] = {"modelId": "anthropic.claude-3", "messages": []}
    callback = _ConverseStreamCallback(cast(Any, span), cast(Any, request))

    callback({"messageStart": {"role": "assistant"}})
    callback(
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": "Thinking "}},
            }
        }
    )
    callback(
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": "it through."}},
            }
        }
    )
    callback(
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"signature": "sig-xyz"}},
            }
        }
    )
    callback(
        {
            "contentBlockDelta": {
                "contentBlockIndex": 1,
                "delta": {"text": "Final answer."},
            }
        }
    )
    callback({"messageStop": {"stopReason": "end_turn"}})
    callback(StopIteration())

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    assert attributes[f"llm.output_messages.0.message.contents.0.{MESSAGE_CONTENT_TYPE}"] == (
        "reasoning"
    )
    assert (
        attributes[f"llm.output_messages.0.message.contents.0.{MESSAGE_CONTENT_TEXT}"]
        == "Thinking it through."
    )
    assert (
        attributes[f"llm.output_messages.0.message.contents.0.{MESSAGE_CONTENT_SIGNATURE}"]
        == "sig-xyz"
    )
    assert attributes[f"llm.output_messages.0.message.contents.1.{MESSAGE_CONTENT_TYPE}"] == "text"
    assert (
        attributes[f"llm.output_messages.0.message.contents.1.{MESSAGE_CONTENT_TEXT}"]
        == "Final answer."
    )
    assert f"llm.output_messages.0.message.contents.0.{MESSAGE_CONTENT_ID}" not in attributes


def test_converse_stream_callback_accumulates_redacted_reasoning_bytes(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    tracer = tracer_provider.get_tracer(__name__)
    span = tracer.start_span("converse.stream.redacted_reasoning")
    request: Dict[str, Any] = {"modelId": "anthropic.claude-3", "messages": []}
    callback = _ConverseStreamCallback(cast(Any, span), cast(Any, request))

    redacted_bytes = b"redacted-bytes"
    callback({"messageStart": {"role": "assistant"}})
    callback(
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"redactedContent": redacted_bytes}},
            }
        }
    )
    callback({"messageStop": {"stopReason": "end_turn"}})
    callback(StopIteration())

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    assert attributes[f"llm.output_messages.0.message.contents.0.{MESSAGE_CONTENT_TYPE}"] == (
        "reasoning"
    )
    assert attributes[
        f"llm.output_messages.0.message.contents.0.{MESSAGE_CONTENT_DATA}"
    ] == base64.b64encode(redacted_bytes).decode("utf-8")
    assert f"llm.output_messages.0.message.contents.0.{MESSAGE_CONTENT_ID}" not in attributes


def test_converse_stream_callback_accumulates_redacted_content_across_multiple_chunks(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    tracer = tracer_provider.get_tracer(__name__)
    span = tracer.start_span("converse.stream.redacted_reasoning_multi_chunk")
    request: Dict[str, Any] = {"modelId": "anthropic.claude-3", "messages": []}
    callback = _ConverseStreamCallback(cast(Any, span), cast(Any, request))

    chunk_1, chunk_2 = b"redacted-", b"bytes-split"
    callback({"messageStart": {"role": "assistant"}})
    callback(
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"redactedContent": chunk_1}},
            }
        }
    )
    callback(
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"redactedContent": chunk_2}},
            }
        }
    )
    callback({"messageStop": {"stopReason": "end_turn"}})
    callback(StopIteration())

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    assert attributes[
        f"llm.output_messages.0.message.contents.0.{MESSAGE_CONTENT_DATA}"
    ] == base64.b64encode(chunk_1 + chunk_2).decode("utf-8")
    assert f"llm.output_messages.0.message.contents.0.{MESSAGE_CONTENT_ID}" not in attributes


def test_converse_stream_callback_reasoning_then_tool_use(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Realistic Claude-on-Bedrock pattern: think, then call a tool."""
    tracer = tracer_provider.get_tracer(__name__)
    span = tracer.start_span("converse.stream.reasoning_then_tool_use")
    request: Dict[str, Any] = {"modelId": "anthropic.claude-3", "messages": []}
    callback = _ConverseStreamCallback(cast(Any, span), cast(Any, request))

    callback({"messageStart": {"role": "assistant"}})
    callback(
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": "I should call the tool."}},
            }
        }
    )
    callback(
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"signature": "sig-1"}},
            }
        }
    )
    callback(
        {
            "contentBlockStart": {
                "contentBlockIndex": 1,
                "start": {"toolUse": {"toolUseId": "tool-1", "name": "lookup"}},
            }
        }
    )
    callback(
        {
            "contentBlockDelta": {
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '{"q": "x"}'}},
            }
        }
    )
    callback({"messageStop": {"stopReason": "tool_use"}})
    callback(StopIteration())

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    assert (
        attributes[f"llm.output_messages.0.message.contents.0.{MESSAGE_CONTENT_TYPE}"]
        == "reasoning"
    )
    assert (
        attributes[f"llm.output_messages.0.message.contents.0.{MESSAGE_CONTENT_TEXT}"]
        == "I should call the tool."
    )
    assert (
        attributes[f"llm.output_messages.0.message.contents.0.{MESSAGE_CONTENT_SIGNATURE}"]
        == "sig-1"
    )
    assert attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"] == (
        "lookup"
    )
    assert f"llm.output_messages.0.message.contents.0.{MESSAGE_CONTENT_ID}" not in attributes
