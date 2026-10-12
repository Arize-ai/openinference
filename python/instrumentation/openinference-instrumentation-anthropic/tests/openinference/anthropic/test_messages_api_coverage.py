"""Tests for coverage of the current Messages API: robust extraction of input messages, streams
whose events fail to accumulate, and every content block type in input and output messages."""

import json
from typing import Any, Dict, List

import httpx2
import pytest
from anthropic import Anthropic
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.anthropic._wrappers import _get_llm_input_messages
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolAttributes,
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
MESSAGE_CONTENT_SIGNATURE = MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE
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
    # a block that cannot be mapped is recorded generically, so that no index is skipped
    assert attributes == {
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}": "system",
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}": "text",
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}": '{"type": "text"}',
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}": "text",
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TEXT}": "be brief",
        f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}": "assistant",
        f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}": "text",
        f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}": '{"type": "text"}',
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


# A response with a block of every type that is not plain text, reasoning, or a client tool call.
_SERVER_TOOL_BLOCKS: List[Dict[str, Any]] = [
    {"type": "text", "text": "Searching."},
    {
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {"query": "weather in Paris"},
    },
    {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": [
            {
                "type": "web_search_result",
                "url": "https://example.com/paris",
                "title": "Paris weather",
                "encrypted_content": "abc",
            }
        ],
    },
    {
        "type": "mcp_tool_use",
        "id": "mcptoolu_1",
        "name": "echo",
        "server_name": "echo-server",
        "input": {"text": "hi"},
    },
    {
        "type": "mcp_tool_result",
        "tool_use_id": "mcptoolu_1",
        "is_error": False,
        "content": [{"type": "text", "text": "hi"}],
    },
    {"type": "mcp_tool_listing", "mcp_server_name": "echo-server", "tools": []},
    {"type": "compaction", "content": "The user asked about Paris.", "signature": "sig"},
    {"type": "container_upload", "file_id": "file_1"},
    {"type": "brand_new_block", "foo": 1},
]


def _assert_server_tool_blocks(attributes: Dict[str, Any], message_prefix: str) -> None:
    contents = f"{message_prefix}.{MESSAGE_CONTENTS}"
    tool_calls = f"{message_prefix}.{MESSAGE_TOOL_CALLS}"
    assert attributes.pop(f"{contents}.0.{MESSAGE_CONTENT_TYPE}") == "text"
    assert attributes.pop(f"{contents}.0.{MESSAGE_CONTENT_TEXT}") == "Searching."
    # server and MCP tool calls are tool calls, in the order of their blocks
    for content_index, tool_index, tool_call_id, name, arguments in [
        (1, 0, "srvtoolu_1", "web_search", {"query": "weather in Paris"}),
        (3, 1, "mcptoolu_1", "echo", {"text": "hi"}),
    ]:
        for prefix in [f"{tool_calls}.{tool_index}", f"{contents}.{content_index}"]:
            assert attributes.pop(f"{prefix}.{TOOL_CALL_ID}") == tool_call_id
            assert attributes.pop(f"{prefix}.{TOOL_CALL_FUNCTION_NAME}") == name
            assert (
                json.loads(attributes.pop(f"{prefix}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"))
                == arguments
            )
        assert attributes.pop(f"{contents}.{content_index}.{MESSAGE_CONTENT_TYPE}") == "tool_use"
    # a compaction block's summary is text
    assert attributes.pop(f"{contents}.6.{MESSAGE_CONTENT_TYPE}") == "text"
    assert attributes.pop(f"{contents}.6.{MESSAGE_CONTENT_TEXT}") == "The user asked about Paris."
    assert attributes.pop(f"{contents}.6.{MESSAGE_CONTENT_SIGNATURE}") == "sig"
    # every other block is recorded generically, as its type and JSON
    for content_index in [2, 4, 5, 7, 8]:
        block = _SERVER_TOOL_BLOCKS[content_index]
        assert attributes.pop(f"{contents}.{content_index}.{MESSAGE_CONTENT_TYPE}") == block["type"]
        recorded = json.loads(attributes.pop(f"{contents}.{content_index}.{MESSAGE_CONTENT_TEXT}"))
        assert {key: recorded[key] for key in block} == block


def _pop_output_message(attributes: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: attributes.pop(key)
        for key in list(attributes)
        if key.startswith(f"{LLM_OUTPUT_MESSAGES}.")
    }


def test_output_server_tool_blocks(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = _mock_anthropic_client(
        lambda request: httpx2.Response(200, json=_message_json(_SERVER_TOOL_BLOCKS))
    )
    client.beta.messages.create(
        model=MODEL, max_tokens=100, messages=[{"role": "user", "content": "weather?"}]
    )
    output_message = _pop_output_message(dict(_get_span(in_memory_span_exporter).attributes or {}))
    assert output_message.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    _assert_server_tool_blocks(output_message, f"{LLM_OUTPUT_MESSAGES}.0")
    assert not output_message


def _content_block_events(index: int, block: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "input" not in block:
        return [
            {"type": "content_block_start", "index": index, "content_block": block},
            {"type": "content_block_stop", "index": index},
        ]
    # tool inputs are streamed as JSON deltas
    partial_json = json.dumps(block["input"])
    return [
        {"type": "content_block_start", "index": index, "content_block": {**block, "input": {}}},
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json[:5]},
        },
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json[5:]},
        },
        {"type": "content_block_stop", "index": index},
    ]


def _server_tool_blocks_stream_handler(request: Any) -> Any:
    return _event_stream_response(
        [
            {"type": "message_start", "message": _message_json([])},
            *(
                event
                for index, block in enumerate(_SERVER_TOOL_BLOCKS)
                for event in _content_block_events(index, block)
            ),
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 9},
            },
            {"type": "message_stop"},
        ]
    )


def test_streaming_output_server_tool_blocks(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = _mock_anthropic_client(_server_tool_blocks_stream_handler)
    stream = client.beta.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": "weather?"}],
        stream=True,
    )
    for _ in stream:
        pass
    output_message = _pop_output_message(dict(_get_span(in_memory_span_exporter).attributes or {}))
    assert output_message.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    _assert_server_tool_blocks(output_message, f"{LLM_OUTPUT_MESSAGES}.0")
    assert not output_message


def test_streaming_manager_output_server_tool_blocks(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = _mock_anthropic_client(_server_tool_blocks_stream_handler)
    with client.beta.messages.stream(
        model=MODEL, max_tokens=100, messages=[{"role": "user", "content": "weather?"}]
    ) as stream:
        stream.until_done()
    output_message = _pop_output_message(dict(_get_span(in_memory_span_exporter).attributes or {}))
    assert output_message.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    _assert_server_tool_blocks(output_message, f"{LLM_OUTPUT_MESSAGES}.0")
    assert not output_message


@pytest.mark.parametrize("as_models", [False, True])
def test_input_server_tool_blocks(as_models: bool) -> None:
    content: List[Any] = _SERVER_TOOL_BLOCKS
    if as_models:
        # e.g. a response passed back as input
        from anthropic._models import construct_type
        from anthropic.types.beta import BetaMessage

        message: Any = construct_type(type_=BetaMessage, value=_message_json(content))
        content = message.content
    attributes = dict(
        _get_llm_input_messages(
            [{"role": "user", "content": "weather?"}, {"role": "assistant", "content": content}]
        )
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "weather?"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "assistant"
    _assert_server_tool_blocks(attributes, f"{LLM_INPUT_MESSAGES}.1")
    assert not attributes


def test_input_user_blocks() -> None:
    contents = f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}"
    blocks: List[Dict[str, Any]] = [
        {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
        {"type": "image", "source": {"type": "file", "file_id": "file_1"}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
        {
            "type": "search_result",
            "source": "https://example.com",
            "title": "Example",
            "content": [{"type": "text", "text": "result"}],
        },
        {"type": "tool_removal", "tool": {"type": "tool_reference", "name": "f"}},
    ]
    attributes = dict(_get_llm_input_messages([{"role": "user", "content": blocks}]))
    image_url = f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
    assert attributes == {
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}": "user",
        f"{contents}.0.{MESSAGE_CONTENT_TYPE}": "image",
        f"{contents}.0.{image_url}": "https://example.com/a.png",
        # a file has no URL, so it is referenced by its id
        f"{contents}.1.{MESSAGE_CONTENT_TYPE}": "image",
        f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_ID}": "file_1",
        f"{contents}.2.{MESSAGE_CONTENT_TYPE}": "image",
        f"{contents}.2.{image_url}": "data:image/png;base64,AAA",
        f"{contents}.3.{MESSAGE_CONTENT_TYPE}": "search_result",
        f"{contents}.3.{MESSAGE_CONTENT_TEXT}": json.dumps(blocks[3]),
        f"{contents}.4.{MESSAGE_CONTENT_TYPE}": "tool_removal",
        f"{contents}.4.{MESSAGE_CONTENT_TEXT}": json.dumps(blocks[4]),
    }


def test_inline_tool_definitions_are_recorded_as_tools(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    tool = {"name": "f", "input_schema": {"type": "object"}}
    inline_tool = {"name": "g", "input_schema": {"type": "object"}}
    tool_addition = {
        "type": "tool_addition",
        "tool": {"type": "tool_definition", "definition": inline_tool},
    }
    client = _mock_anthropic_client(
        lambda request: httpx2.Response(200, json=_message_json([{"type": "text", "text": "ok"}]))
    )
    messages: Any = [
        {"role": "user", "content": "hi"},
        {
            "role": "user",
            "content": [
                tool_addition,
                # a reference to a tool defines nothing new
                {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "f"}},
            ],
        },
    ]
    tools: Any = [tool]
    client.beta.messages.create(model=MODEL, max_tokens=100, tools=tools, messages=messages)
    attributes = dict(_get_span(in_memory_span_exporter).attributes or {})
    llm_tools = SpanAttributes.LLM_TOOLS
    json_schema = ToolAttributes.TOOL_JSON_SCHEMA
    assert json.loads(str(attributes.pop(f"{llm_tools}.0.{json_schema}"))) == tool
    assert json.loads(str(attributes.pop(f"{llm_tools}.1.{json_schema}"))) == inline_tool
    assert f"{llm_tools}.2.{json_schema}" not in attributes
    contents = f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}"
    assert attributes.pop(f"{contents}.0.{MESSAGE_CONTENT_TYPE}") == "tool_addition"
    assert json.loads(str(attributes.pop(f"{contents}.0.{MESSAGE_CONTENT_TEXT}"))) == (
        tool_addition
    )
