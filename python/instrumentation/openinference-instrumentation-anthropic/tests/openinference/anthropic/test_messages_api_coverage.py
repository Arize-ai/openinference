"""Tests for coverage of the current Messages API: robust extraction of input messages, streams
whose events fail to accumulate, and every content block type in input and output messages."""

import json
import warnings
from typing import Any, Callable, Dict, Iterator, List

import httpx2
import pytest
from anthropic import Anthropic, AsyncAnthropic
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.anthropic._content import _get_response_metadata
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


@pytest.mark.parametrize("streaming", [False, True])
def test_unknown_blocks_do_not_warn(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
    streaming: bool,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if streaming:
            client = _mock_anthropic_client(_server_tool_blocks_stream_handler)
            stream = client.beta.messages.create(
                model=MODEL,
                max_tokens=100,
                messages=[{"role": "user", "content": "weather?"}],
                stream=True,
            )
            for _ in stream:
                pass
        else:
            response = _message_json(_SERVER_TOOL_BLOCKS)
            client = _mock_anthropic_client(lambda request: httpx2.Response(200, json=response))
            client.beta.messages.create(
                model=MODEL, max_tokens=100, messages=[{"role": "user", "content": "weather?"}]
            )
    attributes = dict(_get_span(in_memory_span_exporter).attributes or {})
    output = json.loads(str(attributes[SpanAttributes.OUTPUT_VALUE]))
    unknown_block = output["content"][-1]
    assert (unknown_block["type"], unknown_block["foo"]) == ("brand_new_block", 1)
    assert f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}" in attributes


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
        # a file has no URL, so the block is recorded as JSON
        f"{contents}.1.{MESSAGE_CONTENT_TYPE}": "image",
        f"{contents}.1.{MESSAGE_CONTENT_TEXT}": json.dumps(blocks[1]),
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


def test_tool_changes_are_applied_to_tools(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    tool_f = {"name": "f", "input_schema": {"type": "object"}}
    tool_h = {"name": "h", "input_schema": {"type": "object"}}
    toolset = {"type": "mcp_toolset", "mcp_server_name": "echo-server"}
    inline_tool = {"name": "g", "input_schema": {"type": "object"}}
    client = _mock_anthropic_client(
        lambda request: httpx2.Response(200, json=_message_json([{"type": "text", "text": "ok"}]))
    )
    # tool changes are sent in mid-conversation system messages
    messages: Any = [
        {"role": "user", "content": "hi"},
        {
            "role": "system",
            "content": [
                {
                    "type": "tool_addition",
                    "tool": {"type": "tool_definition", "definition": inline_tool},
                },
                {"type": "tool_removal", "tool": {"type": "tool_reference", "name": "f"}},
                {"type": "tool_removal", "tool": {"type": "tool_reference", "name": "g"}},
                {
                    "type": "tool_removal",
                    "tool": {"type": "mcp_toolset_reference", "server_name": "echo-server"},
                },
            ],
        },
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "again"},
        {
            "role": "system",
            "content": [
                {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "f"}},
                {
                    "type": "tool_addition",
                    "tool": {"type": "tool_definition", "definition": inline_tool},
                },
            ],
        },
    ]
    tools: Any = [tool_f, tool_h, toolset]
    client.beta.messages.create(model=MODEL, max_tokens=100, tools=tools, messages=messages)
    attributes = dict(_get_span(in_memory_span_exporter).attributes or {})
    llm_tools = SpanAttributes.LLM_TOOLS
    json_schema = ToolAttributes.TOOL_JSON_SCHEMA
    recorded = []
    index = 0
    while (key := f"{llm_tools}.{index}.{json_schema}") in attributes:
        recorded.append(json.loads(str(attributes[key])))
        index += 1
    # the removed toolset is gone, and the re-added tools are listed once each
    assert recorded == [tool_h, tool_f, inline_tool]


_CITATION = {
    "type": "web_search_result_location",
    "cited_text": "Sunny.",
    "url": "https://example.com/paris",
    "title": "Paris weather",
    "encrypted_index": "abc",
}
_ITERATIONS = [
    {
        "type": "compaction",
        "input_tokens": 1000,
        "output_tokens": 50,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    },
    {
        "type": "message",
        "input_tokens": 60,
        "output_tokens": 20,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    },
]
# Provider-specific data with no semantic convention of its own, which is recorded as metadata.
_RESPONSE_DATA: Dict[str, Any] = {
    "stop_details": {"type": "refusal", "category": "cyber", "explanation": "No."},
    "diagnostics": {"cache_miss_reason": {"type": "system_changed"}},
    "container": {"id": "container_1", "expires_at": "2026-10-12T00:00:00Z"},
    "context_management": {
        "applied_edits": [
            {
                "type": "clear_tool_uses_20250919",
                "cleared_input_tokens": 100,
                "cleared_tool_uses": 2,
            }
        ]
    },
    "input_transformations": [
        {
            "type": "thinking_dropped",
            "path": "messages.1.content.0",
            "reason": "model_binding_mismatch",
        }
    ],
}
_RESPONSE_CONTENT: List[Dict[str, Any]] = [
    {"type": "thinking", "thinking": "Hmm.", "signature": "sig"},
    {"type": "text", "text": "Sunny.", "citations": [_CITATION]},
    {
        "type": "mcp_tool_use",
        "id": "mcptoolu_1",
        "name": "echo",
        "server_name": "echo-server",
        "input": {},
    },
]
_RESPONSE_USAGE: Dict[str, Any] = {
    "input_tokens": 60,
    "output_tokens": 20,
    "output_tokens_details": {"thinking_tokens": 7},
    "iterations": _ITERATIONS,
    "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 0},
}


def _assert_response_metadata_and_usage(attributes: Dict[str, Any]) -> None:
    assert json.loads(str(attributes[SpanAttributes.METADATA])) == {
        # the context's metadata is kept
        "user_key": "user_value",
        "stop_details": _RESPONSE_DATA["stop_details"],
        "cache_miss_reason": {"type": "system_changed"},
        "container_id": "container_1",
        "context_management_applied_edits": _RESPONSE_DATA["context_management"]["applied_edits"],
        "input_transformations": _RESPONSE_DATA["input_transformations"],
        "usage_iterations": _ITERATIONS,
        "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 0},
        "citations": {"1": [_CITATION]},
        "mcp_server_names": {"mcptoolu_1": "echo-server"},
    }
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING] == 7
    # a compaction iteration's tokens are not in the top-level usage, so not in the counts
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 60
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 20
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 80


def test_response_metadata_and_usage(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    from openinference.instrumentation import using_metadata

    response = _message_json(_RESPONSE_CONTENT, **_RESPONSE_DATA, usage=_RESPONSE_USAGE)
    client = _mock_anthropic_client(lambda request: httpx2.Response(200, json=response))
    with using_metadata({"user_key": "user_value"}):
        client.beta.messages.create(
            model=MODEL, max_tokens=100, messages=[{"role": "user", "content": "weather?"}]
        )
    _assert_response_metadata_and_usage(dict(_get_span(in_memory_span_exporter).attributes or {}))


def _response_metadata_stream_handler(request: Any) -> Any:
    return _event_stream_response(
        [
            {
                "type": "message_start",
                "message": _message_json(
                    [],
                    diagnostics=_RESPONSE_DATA["diagnostics"],
                    input_transformations=_RESPONSE_DATA["input_transformations"],
                ),
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Hmm."},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Sunny."},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "citations_delta", "citation": _CITATION},
            },
            {"type": "content_block_stop", "index": 1},
            *_content_block_events(2, _RESPONSE_CONTENT[2]),
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "refusal",
                    "stop_sequence": None,
                    "stop_details": _RESPONSE_DATA["stop_details"],
                    "container": _RESPONSE_DATA["container"],
                },
                "context_management": _RESPONSE_DATA["context_management"],
                "usage": _RESPONSE_USAGE,
            },
            {"type": "message_stop"},
        ]
    )


def test_streaming_response_metadata_and_usage(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    from openinference.instrumentation import using_metadata

    client = _mock_anthropic_client(_response_metadata_stream_handler)
    with using_metadata({"user_key": "user_value"}):
        stream = client.beta.messages.create(
            model=MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": "weather?"}],
            stream=True,
        )
    # finished outside of the context the span was started in
    for _ in stream:
        pass
    _assert_response_metadata_and_usage(dict(_get_span(in_memory_span_exporter).attributes or {}))


def test_streaming_manager_response_metadata_and_usage(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    from openinference.instrumentation import using_metadata

    client = _mock_anthropic_client(_response_metadata_stream_handler)
    with using_metadata({"user_key": "user_value"}):
        manager = client.beta.messages.stream(
            model=MODEL, max_tokens=100, messages=[{"role": "user", "content": "weather?"}]
        )
    with manager as stream:
        stream.until_done()
    _assert_response_metadata_and_usage(dict(_get_span(in_memory_span_exporter).attributes or {}))


def test_no_metadata_without_response_data(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    from openinference.instrumentation import using_metadata

    client = _mock_anthropic_client(
        lambda request: httpx2.Response(200, json=_message_json([{"type": "text", "text": "ok"}]))
    )
    with using_metadata({"user_key": "user_value"}):
        client.messages.create(
            model=MODEL, max_tokens=100, messages=[{"role": "user", "content": "hi"}]
        )
    attributes = dict(_get_span(in_memory_span_exporter).attributes or {})
    assert json.loads(str(attributes[SpanAttributes.METADATA])) == {"user_key": "user_value"}
    assert SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING not in attributes


def _platform_clients(handler: Any) -> Dict[str, Any]:
    from anthropic import AnthropicBedrock, AnthropicFoundry, AnthropicVertex

    def http_client() -> Any:
        return httpx2.Client(transport=httpx2.MockTransport(handler))

    return {
        "anthropic": _mock_anthropic_client(handler),
        "aws": AnthropicBedrock(api_key="k", aws_region="us-east-1", http_client=http_client()),
        "google": AnthropicVertex(
            region="us-east5", project_id="p", access_token="t", http_client=http_client()
        ),
        "azure": AnthropicFoundry(api_key="k", resource="r", http_client=http_client()),
    }


@pytest.mark.parametrize("provider", ["anthropic", "aws", "google", "azure"])
def test_provider_is_the_platform_hosting_the_model(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
    provider: str,
) -> None:
    client = _platform_clients(
        lambda request: httpx2.Response(200, json=_message_json([{"type": "text", "text": "ok"}]))
    )[provider]
    client.messages.create(
        model=MODEL, max_tokens=100, messages=[{"role": "user", "content": "hi"}]
    )
    attributes = dict(_get_span(in_memory_span_exporter).attributes or {})
    assert attributes[SpanAttributes.LLM_PROVIDER] == provider
    assert attributes[SpanAttributes.LLM_SYSTEM] == "anthropic"


def _text_message_handler(request: Any) -> Any:
    # Vertex AI's path is count-tokens:rawPredict
    if "count_tokens" in request.url.path or "count-tokens" in request.url.path:
        if "beta=true" in str(request.url):
            return httpx2.Response(
                200, json={"input_tokens": 12, "context_management": _COUNT_CONTEXT_MANAGEMENT}
            )
        return httpx2.Response(200, json={"input_tokens": 12})
    if b'"stream": true' in request.content or b'"stream":true' in request.content:
        return _event_stream_response(
            [
                {"type": "message_start", "message": _message_json([])},
                *_content_block_events(0, {"type": "text", "text": "ok"}),
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
                {"type": "message_stop"},
            ]
        )
    return httpx2.Response(200, json=_message_json([{"type": "text", "text": "ok"}]))


@pytest.mark.parametrize("provider", ["aws", "google"])
def test_platform_beta_messages_are_traced(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
    provider: str,
) -> None:
    client = _platform_clients(_text_message_handler)[provider]
    messages: Any = [{"role": "user", "content": "hi"}]
    client.beta.messages.create(model=MODEL, max_tokens=100, messages=messages)
    names = ["beta.messages.create"]
    if provider == "google":
        # Bedrock streams AWS event streams, which need botocore to decode
        with client.beta.messages.stream(model=MODEL, max_tokens=100, messages=messages) as stream:
            stream.until_done()
        names.append("beta.messages.stream")
    # one span each, so none is traced twice
    spans = in_memory_span_exporter.get_finished_spans()
    assert [span.name for span in spans] == names
    output_text = f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"
    for span in spans:
        attributes = dict(span.attributes or {})
        assert attributes[SpanAttributes.LLM_PROVIDER] == provider
        assert attributes[f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}"] == "hi"
        assert attributes[output_text] == "ok"


async def test_platform_async_beta_messages_are_traced(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    from anthropic import AsyncAnthropicBedrock

    # Any, since the SDK types the methods it copies as the first-party resource's
    client: Any = AsyncAnthropicBedrock(
        api_key="k",
        aws_region="us-east-1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(_text_message_handler)),
    )
    await client.beta.messages.create(
        model=MODEL, max_tokens=100, messages=[{"role": "user", "content": "hi"}]
    )
    span = _get_span(in_memory_span_exporter)
    assert span.name == "beta.messages.create"
    assert dict(span.attributes or {})[SpanAttributes.LLM_PROVIDER] == "aws"


_COUNT_CONTEXT_MANAGEMENT = {"original_input_tokens": 50}


def _assert_count_tokens_span(span: ReadableSpan, name: str, provider: str = "anthropic") -> None:
    assert span.name == name
    attributes = dict(span.attributes or {})
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == provider
    assert attributes.pop(SpanAttributes.LLM_SYSTEM) == "anthropic"
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == MODEL
    assert attributes.pop(SpanAttributes.LLM_REQUEST_MODEL_NAME) == MODEL
    assert json.loads(str(attributes.pop(SpanAttributes.INPUT_VALUE)))["model"] == MODEL
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "system"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "be brief"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENT}") == "hi"
    assert json.loads(
        str(attributes.pop(f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}"))
    ) == {"name": "f", "input_schema": {"type": "object"}}
    assert json.loads(str(attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS))) == {
        "tool_choice": {"type": "auto"}
    }
    # the count is the output, and nothing is generated, so there are no token counts
    expected_output: Dict[str, Any] = {"input_tokens": 12}
    if name.startswith("beta.") and provider == "anthropic":
        expected_output["context_management"] = _COUNT_CONTEXT_MANAGEMENT
    assert json.loads(str(attributes.pop(SpanAttributes.OUTPUT_VALUE))) == expected_output
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert not attributes
    assert span.status.is_ok


_COUNT_TOKENS_KWARGS: Dict[str, Any] = {
    "model": MODEL,
    "system": "be brief",
    "messages": [{"role": "user", "content": "hi"}],
    "tools": [{"name": "f", "input_schema": {"type": "object"}}],
    "tool_choice": {"type": "auto"},
}


def test_count_tokens_span_has_context_attributes(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    from openinference.instrumentation import using_attributes

    client = _mock_anthropic_client(_text_message_handler)
    with using_attributes(session_id="session-1", user_id="user-1", metadata={"k": "v"}):
        client.messages.count_tokens(**_COUNT_TOKENS_KWARGS)
    attributes = dict(_get_span(in_memory_span_exporter).attributes or {})
    assert attributes[SpanAttributes.SESSION_ID] == "session-1"
    assert attributes[SpanAttributes.USER_ID] == "user-1"
    assert json.loads(str(attributes[SpanAttributes.METADATA])) == {"k": "v"}


def test_platform_beta_count_tokens(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = _platform_clients(_text_message_handler)["google"]
    assert client.beta.messages.count_tokens(**_COUNT_TOKENS_KWARGS).input_tokens == 12
    _assert_count_tokens_span(
        _get_span(in_memory_span_exporter), "beta.messages.count_tokens", provider="google"
    )


def test_raw_count_tokens_response(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = _mock_anthropic_client(_text_message_handler)
    response = client.messages.with_raw_response.count_tokens(**_COUNT_TOKENS_KWARGS)
    assert response.parse().input_tokens == 12
    _assert_count_tokens_span(_get_span(in_memory_span_exporter), "messages.count_tokens")


def test_suppressed_count_tokens_is_not_traced(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    from openinference.instrumentation import suppress_tracing

    client = _mock_anthropic_client(_text_message_handler)
    with suppress_tracing():
        client.messages.count_tokens(**_COUNT_TOKENS_KWARGS)
    assert not in_memory_span_exporter.get_finished_spans()


def test_uninstrumentation_restores_count_tokens_and_platform_beta_messages(
    tracer_provider: TracerProvider,
) -> None:
    from anthropic.lib.bedrock import _beta_messages as bedrock_beta_messages
    from anthropic.lib.vertex import _beta_messages as vertex_beta_messages
    from anthropic.resources.beta.messages import AsyncMessages as AsyncBetaMessages
    from anthropic.resources.beta.messages import Messages as BetaMessages
    from anthropic.resources.messages import AsyncMessages, Messages
    from wrapt import FunctionWrapper

    from openinference.instrumentation.anthropic import AnthropicInstrumentor

    wrapped = [
        (Messages, "count_tokens"),
        (AsyncMessages, "count_tokens"),
        (BetaMessages, "count_tokens"),
        (AsyncBetaMessages, "count_tokens"),
        *(
            (cls, method)
            for module in [bedrock_beta_messages, vertex_beta_messages]
            for cls in [module.Messages, module.AsyncMessages]
            for method in ["create", "parse", "stream"]
        ),
        (vertex_beta_messages.Messages, "count_tokens"),
        (vertex_beta_messages.AsyncMessages, "count_tokens"),
    ]
    originals = {(cls, method): cls.__dict__[method] for cls, method in wrapped}
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
    try:
        for cls, method in wrapped:
            assert isinstance(cls.__dict__[method], FunctionWrapper), (cls, method)
    finally:
        AnthropicInstrumentor().uninstrument()
    for cls, method in wrapped:
        assert cls.__dict__[method] is originals[(cls, method)]


@pytest.mark.parametrize("accessor", ["with_raw_response", "with_streaming_response"])
def test_uninstrumentation_stops_cached_accessors(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    accessor: str,
) -> None:
    from openinference.instrumentation.anthropic import AnthropicInstrumentor

    client = _mock_anthropic_client(_text_message_handler)
    kwargs: Dict[str, Any] = dict(
        model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "hi"}]
    )

    def call() -> None:
        if accessor == "with_raw_response":
            client.messages.with_raw_response.create(**kwargs).parse()
        else:
            with client.messages.with_streaming_response.create(**kwargs) as response:
                response.parse()

    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
    try:
        call()  # caches the accessor while instrumented
    finally:
        AnthropicInstrumentor().uninstrument()
    assert len(in_memory_span_exporter.get_finished_spans()) == 1
    call()
    assert len(in_memory_span_exporter.get_finished_spans()) == 1
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
    try:
        call()  # the same cached accessor, traced again and only once
    finally:
        AnthropicInstrumentor().uninstrument()
    assert len(in_memory_span_exporter.get_finished_spans()) == 2


@pytest.mark.parametrize("accessor", ["with_raw_response", "with_streaming_response"])
async def test_uninstrumentation_stops_async_cached_accessors(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    accessor: str,
) -> None:
    from openinference.instrumentation.anthropic import AnthropicInstrumentor

    client = AsyncAnthropic(
        api_key="sk-ant-fake",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(_text_message_handler)),
    )
    kwargs: Dict[str, Any] = dict(
        model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "hi"}]
    )

    async def call() -> None:
        if accessor == "with_raw_response":
            await (await client.messages.with_raw_response.create(**kwargs)).parse()
        else:
            async with client.messages.with_streaming_response.create(**kwargs) as response:
                await response.parse()

    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
    try:
        await call()  # caches the accessor while instrumented
    finally:
        AnthropicInstrumentor().uninstrument()
    assert len(in_memory_span_exporter.get_finished_spans()) == 1
    await call()
    assert len(in_memory_span_exporter.get_finished_spans()) == 1
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
    try:
        await call()  # the same cached accessor, traced again and only once
    finally:
        AnthropicInstrumentor().uninstrument()
    assert len(in_memory_span_exporter.get_finished_spans()) == 2


def test_reinstrumentation_sends_cached_accessor_spans_to_the_new_tracer_provider() -> None:
    from openinference.instrumentation.anthropic import AnthropicInstrumentor

    client = _mock_anthropic_client(_text_message_handler)
    kwargs: Dict[str, Any] = dict(
        model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "hi"}]
    )
    exporters = []
    for _ in range(2):
        exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
        exporters.append(exporter)
        AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
        try:
            client.messages.with_raw_response.create(**kwargs).parse()
        finally:
            AnthropicInstrumentor().uninstrument()
    assert [len(exporter.get_finished_spans()) for exporter in exporters] == [1, 1]


@pytest.fixture()
def instrument_with_config(tracer_provider: TracerProvider) -> Iterator[Callable[..., None]]:
    from openinference.instrumentation import TraceConfig
    from openinference.instrumentation.anthropic import AnthropicInstrumentor

    def instrument(**config: Any) -> None:
        AnthropicInstrumentor().instrument(
            tracer_provider=tracer_provider, config=TraceConfig(**config)
        )

    yield instrument
    AnthropicInstrumentor().uninstrument()


@pytest.mark.parametrize(
    "config, hides_citations, hides_cited_text, hides_explanation",
    [
        ({}, False, False, False),
        ({"hide_inputs": True}, True, True, False),
        ({"hide_input_messages": True}, True, True, False),
        ({"hide_input_text": True}, False, True, False),
        ({"hide_outputs": True}, True, True, True),
        ({"hide_output_messages": True}, True, True, True),
        ({"hide_output_text": True}, False, True, True),
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_metadata_respects_trace_config(
    in_memory_span_exporter: InMemorySpanExporter,
    instrument_with_config: Callable[..., None],
    config: Dict[str, Any],
    hides_citations: bool,
    hides_cited_text: bool,
    hides_explanation: bool,
    streaming: bool,
) -> None:
    instrument_with_config(**config)
    if streaming:
        client = _mock_anthropic_client(_response_metadata_stream_handler)
        stream = client.beta.messages.create(
            model=MODEL, max_tokens=100, messages=[{"role": "user", "content": "hi"}], stream=True
        )
        for _ in stream:
            pass
    else:
        response = _message_json(_RESPONSE_CONTENT, **_RESPONSE_DATA, usage=_RESPONSE_USAGE)
        client = _mock_anthropic_client(lambda request: httpx2.Response(200, json=response))
        client.beta.messages.create(
            model=MODEL, max_tokens=100, messages=[{"role": "user", "content": "hi"}]
        )
    metadata = json.loads(
        str(dict(_get_span(in_memory_span_exporter).attributes or {})[SpanAttributes.METADATA])
    )
    # the citations name and quote the sources, so they are left out when the input or output
    # messages are hidden, and otherwise only their cited text is when the input or output text is
    if hides_citations:
        assert "citations" not in metadata
    else:
        (citation,) = metadata["citations"]["1"]
        assert ("cited_text" in citation) is not hides_cited_text
        # the citation's other fields are kept
        assert citation == {
            key: value
            for key, value in _CITATION.items()
            if not (hides_cited_text and key == "cited_text")
        }
    # the explanation describes the output
    assert ("explanation" in metadata["stop_details"]) is not hides_explanation
    assert metadata["stop_details"]["category"] == "cyber"
    assert metadata["container_id"] == "container_1"


_IMAGE_DATA = "IMAGE" * 100
_PDF_DATA = "PDF" * 100


def _blocks_with_nested_data() -> List[Dict[str, Any]]:
    image = {"type": "base64", "media_type": "image/png", "data": _IMAGE_DATA}
    return [
        {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": _PDF_DATA},
        },
        {
            "type": "document",
            "source": {
                "type": "content",
                "content": [
                    {"type": "image", "source": image},
                    {"type": "image", "source": {"type": "url", "url": "https://e.com/a.png"}},
                ],
            },
        },
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": [{"type": "image", "source": image}],
        },
    ]


@pytest.mark.parametrize("hide_images", [False, True])
def test_blocks_recorded_as_json_redact_their_data(hide_images: bool) -> None:
    attributes = dict(
        _get_llm_input_messages(
            [{"role": "user", "content": _blocks_with_nested_data()}], hide_images=hide_images
        )
    )
    contents = f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}"
    redacted = "__REDACTED__"
    # base64 data is always redacted, keeping the type and media type
    assert json.loads(str(attributes[f"{contents}.0.{MESSAGE_CONTENT_TEXT}"]))["source"] == {
        "type": "base64",
        "media_type": "application/pdf",
        "data": redacted,
    }
    nested = json.loads(str(attributes[f"{contents}.1.{MESSAGE_CONTENT_TEXT}"]))
    base64_image, url_image = nested["source"]["content"]
    assert base64_image["source"] == {"type": "base64", "media_type": "image/png", "data": redacted}
    # other images are redacted only if images are hidden
    assert url_image["source"] == {
        "type": "url",
        "url": redacted if hide_images else "https://e.com/a.png",
    }
    tool_result = json.loads(str(attributes[f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}"]))
    assert tool_result[0]["source"]["data"] == redacted
    for value in attributes.values():
        assert _IMAGE_DATA not in str(value) and _PDF_DATA not in str(value)


def test_trace_config_hides_images_nested_in_input_blocks(
    in_memory_span_exporter: InMemorySpanExporter,
    instrument_with_config: Callable[..., None],
) -> None:
    instrument_with_config(hide_input_images=True)
    client = _mock_anthropic_client(_text_message_handler)
    messages: Any = [{"role": "user", "content": _blocks_with_nested_data()}]
    client.messages.create(model=MODEL, max_tokens=100, messages=messages)
    attributes = dict(_get_span(in_memory_span_exporter).attributes or {})
    nested = json.loads(
        str(attributes[f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TEXT}"])
    )
    assert nested["source"]["content"][1]["source"]["url"] == "__REDACTED__"


def test_models_are_dumped_without_pydantic_v2_arguments() -> None:
    from openinference.instrumentation.anthropic._content import _to_jsonable

    class V1Model:
        # like the SDK's shim of model_dump on Pydantic v1, which rejects `warnings`
        def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
            if "warnings" in kwargs:
                raise ValueError("warnings is only supported in Pydantic v2")
            return {"type": "text", "text": "hi"}

    assert _to_jsonable([V1Model()]) == [{"type": "text", "text": "hi"}]


def test_bad_block_is_logged_at_debug_level(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.DEBUG):
        dict(_get_llm_input_messages([{"role": "user", "content": [{"type": "text"}]}]))
    records = [record for record in caplog.records if record.msg == "Failed to get attributes."]
    assert records and all(record.levelno == logging.DEBUG for record in records)
    assert all(record.exc_info for record in records)


def test_zero_server_tool_counts_are_not_recorded() -> None:
    usage = {
        "input_tokens": 1,
        "output_tokens": 1,
        "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
    }
    assert "server_tool_use" not in _get_response_metadata({"usage": usage})


_FALLBACK_BLOCK: Dict[str, Any] = {
    "type": "fallback",
    "from": {"model": "claude-fable-5"},
    "to": {"model": "claude-opus-4-8"},
    "trigger": {"type": "refusal", "category": "cyber"},
}


def test_output_value_has_the_api_field_names(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = _mock_anthropic_client(
        lambda request: httpx2.Response(200, json=_message_json([_FALLBACK_BLOCK]))
    )
    client.beta.messages.create(
        model=MODEL, max_tokens=100, messages=[{"role": "user", "content": "hi"}]
    )
    attributes = dict(_get_span(in_memory_span_exporter).attributes or {})
    (block,) = json.loads(str(attributes[SpanAttributes.OUTPUT_VALUE]))["content"]
    assert block["from"] == {"model": "claude-fable-5"}
    assert "from_" not in block


_CODE_EXECUTION_CALLER = {"type": "code_execution_20250825", "tool_id": "srvtoolu_1"}
# the request: a call by code execution, which the response's entry for the same id replaces,
# and its result
_TOOL_CALL_MESSAGES: List[Dict[str, Any]] = [
    {"role": "user", "content": "hi"},
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "f",
                "input": {},
                "caller": _CODE_EXECUTION_CALLER,
                "toolset_name": "old",
            },
            {
                "type": "tool_use",
                "id": "toolu_2",
                "name": "f",
                "input": {},
                "caller": _CODE_EXECUTION_CALLER,
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_2", "toolset_name": "browser"},
        ],
    },
]
# the response: a direct call, which adds no caller, and the replacement for toolu_1
_TOOL_CALL_RESPONSE_BLOCKS: List[Dict[str, Any]] = [
    {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "f",
        "input": {},
        "caller": {"type": "code_execution_20260120", "tool_id": "srvtoolu_2"},
        "toolset_name": "browser",
    },
    {
        "type": "tool_use",
        "id": "toolu_3",
        "name": "click",
        "input": {},
        "caller": {"type": "direct"},
    },
]


def _assert_tool_call_metadata(attributes: Dict[str, Any]) -> None:
    metadata = json.loads(str(attributes[SpanAttributes.METADATA]))
    assert metadata["tool_callers"] == {
        "toolu_1": {"type": "code_execution_20260120", "tool_id": "srvtoolu_2"},
        "toolu_2": _CODE_EXECUTION_CALLER,
    }
    assert metadata["toolset_names"] == {"toolu_1": "browser", "toolu_2": "browser"}
    # the tool call's name is the member's own
    tool_calls = f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}"
    assert attributes[f"{tool_calls}.0.{TOOL_CALL_FUNCTION_NAME}"] == "f"
    assert attributes[f"{tool_calls}.1.{TOOL_CALL_FUNCTION_NAME}"] == "click"


def test_tool_callers_and_toolsets_are_recorded(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    response = _message_json(_TOOL_CALL_RESPONSE_BLOCKS)
    client = _mock_anthropic_client(lambda request: httpx2.Response(200, json=response))
    messages: Any = _TOOL_CALL_MESSAGES
    client.messages.create(model=MODEL, max_tokens=100, messages=messages)
    _assert_tool_call_metadata(dict(_get_span(in_memory_span_exporter).attributes or {}))


def test_streaming_tool_callers_and_toolsets_are_recorded(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    def handler(request: Any) -> Any:
        return _event_stream_response(
            [
                {"type": "message_start", "message": _message_json([])},
                *(
                    event
                    for index, block in enumerate(_TOOL_CALL_RESPONSE_BLOCKS)
                    for event in _content_block_events(index, block)
                ),
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                    "usage": {"output_tokens": 9},
                },
                {"type": "message_stop"},
            ]
        )

    client = _mock_anthropic_client(handler)
    messages: Any = _TOOL_CALL_MESSAGES
    stream = client.messages.create(model=MODEL, max_tokens=100, messages=messages, stream=True)
    for _ in stream:
        pass
    _assert_tool_call_metadata(dict(_get_span(in_memory_span_exporter).attributes or {}))


def test_compaction_blocks_record_encrypted_content_and_their_tools(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    inline_tool = {"name": "g", "input_schema": {"type": "object"}}
    compaction = {
        "type": "compaction",
        "content": "summary",
        "encrypted_content": "opaque",
        "tool_changes": [
            {
                "type": "tool_addition",
                "tool": {"type": "tool_definition", "definition": inline_tool},
            },
            # a reference to a tool defines nothing new
            {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "f"}},
            {"type": "tool_removal", "tool": {"type": "tool_reference", "name": "h"}},
        ],
    }
    response = _message_json([{**compaction, "encrypted_content": "opaque output"}])
    client = _mock_anthropic_client(lambda request: httpx2.Response(200, json=response))
    messages: Any = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [compaction]},
        {"role": "user", "content": "more"},
    ]
    client.beta.messages.create(model=MODEL, max_tokens=100, messages=messages)
    attributes = dict(_get_span(in_memory_span_exporter).attributes or {})
    encrypted_content = MessageContentAttributes.MESSAGE_CONTENT_ENCRYPTED_CONTENT
    for prefix, value in [
        (f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.0", "opaque"),
        (f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0", "opaque output"),
    ]:
        assert attributes[f"{prefix}.{MESSAGE_CONTENT_TYPE}"] == "text"
        assert attributes[f"{prefix}.{MESSAGE_CONTENT_TEXT}"] == "summary"
        assert attributes[f"{prefix}.{encrypted_content}"] == value
    llm_tools = SpanAttributes.LLM_TOOLS
    json_schema = ToolAttributes.TOOL_JSON_SCHEMA
    assert json.loads(str(attributes[f"{llm_tools}.0.{json_schema}"])) == inline_tool
    assert f"{llm_tools}.1.{json_schema}" not in attributes


def test_hidden_file_images_are_redacted() -> None:
    block = {"type": "image", "source": {"type": "file", "file_id": "file_1"}}
    attributes = dict(
        _get_llm_input_messages([{"role": "user", "content": [block]}], hide_images=True)
    )
    contents = f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}"
    assert attributes[f"{contents}.0.{MESSAGE_CONTENT_TYPE}"] == "image"
    assert json.loads(str(attributes[f"{contents}.0.{MESSAGE_CONTENT_TEXT}"])) == {
        "type": "image",
        "source": {"type": "file", "file_id": "__REDACTED__"},
    }
    assert not any(key.endswith(MessageContentAttributes.MESSAGE_CONTENT_ID) for key in attributes)


@pytest.mark.parametrize(
    "index, exception, buffered",
    [
        (1, ValueError("Unable to parse tool parameter JSON from model."), True),
        # not the SDK's JSON parse failure
        (1, RuntimeError("Unexpected event order"), False),
        # a block that does not track tool input
        (0, ValueError("Unable to parse tool parameter JSON from model."), False),
        # no block at the index
        (5, ValueError("Unable to parse tool parameter JSON from model."), False),
    ],
)
def test_only_failed_tool_input_json_is_buffered(
    monkeypatch: pytest.MonkeyPatch, index: int, exception: Exception, buffered: bool
) -> None:
    from anthropic._models import construct_type
    from anthropic.lib.streaming import _messages
    from anthropic.types import RawMessageStreamEvent

    from openinference.instrumentation.anthropic._stream import _MessageResponseAccumulator

    failing_delta = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": '{"a": '},
    }
    original_accumulate_event = _messages.accumulate_event

    def accumulate_event(**kwargs: Any) -> Any:
        if kwargs["event"].type == "content_block_delta":
            raise exception
        return original_accumulate_event(**kwargs)

    monkeypatch.setattr(_messages, "accumulate_event", accumulate_event)
    accumulator = _MessageResponseAccumulator(is_beta=False, request_headers=httpx2.Headers())
    for event in [
        {"type": "message_start", "message": _message_json([])},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}},
        },
        failing_delta,
    ]:
        chunk: Any = construct_type(type_=RawMessageStreamEvent, value=event)
        accumulator.process_chunk(chunk)
    assert (index in accumulator._json_bufs) is buffered
    assert (index in accumulator._stale_inputs) is buffered
    snapshot = accumulator._result()
    assert snapshot.content[0].text == ""
    assert snapshot.content[1].input == {}


@pytest.mark.parametrize("as_model", [False, True])
@pytest.mark.parametrize(
    "trigger, text",
    [
        (
            {"type": "refusal", "category": "cyber"},
            "claude-fable-5 -> claude-opus-4-8 (refusal: cyber)",
        ),
        ({"type": "refusal"}, "claude-fable-5 -> claude-opus-4-8"),
    ],
)
def test_fallback_blocks_are_summarized(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
    as_model: bool,
    trigger: Dict[str, Any],
    text: str,
) -> None:
    block = {**_FALLBACK_BLOCK, "trigger": trigger}
    content: List[Any] = [block]
    if as_model:
        # e.g. a response passed back as input
        from anthropic._models import construct_type
        from anthropic.types.beta import BetaMessage

        message: Any = construct_type(type_=BetaMessage, value=_message_json(content))
        content = message.content
    client = _mock_anthropic_client(
        lambda request: httpx2.Response(200, json=_message_json([block]))
    )
    messages: Any = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": content},
        {"role": "user", "content": "more"},
    ]
    client.beta.messages.create(model=MODEL, max_tokens=100, messages=messages)
    attributes = dict(_get_span(in_memory_span_exporter).attributes or {})
    for prefix in [
        f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.0",
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0",
    ]:
        assert attributes[f"{prefix}.{MESSAGE_CONTENT_TYPE}"] == "fallback"
        assert attributes[f"{prefix}.{MESSAGE_CONTENT_TEXT}"] == text


def test_trace_config_is_passed_to_the_wrappers(tracer_provider: TracerProvider) -> None:
    from anthropic.lib.vertex import _beta_messages as vertex_beta_messages
    from anthropic.resources.messages import Messages

    from openinference.instrumentation import TraceConfig
    from openinference.instrumentation.anthropic import AnthropicInstrumentor

    config = TraceConfig(hide_output_text=True)
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider, config=config)
    try:
        for method in [
            Messages.__dict__["create"],
            Messages.__dict__["stream"],
            Messages.__dict__["count_tokens"],
            vertex_beta_messages.Messages.__dict__["create"],
        ]:
            assert method._self_wrapper._config is config
    finally:
        AnthropicInstrumentor().uninstrument()


def test_wrappers_default_to_a_trace_config() -> None:
    from opentelemetry.trace import NoOpTracer

    from openinference.instrumentation import TraceConfig
    from openinference.instrumentation.anthropic._wrappers import _MessagesWrapper

    # a tracer that is not an OITracer, so has no config of its own
    wrapper = _MessagesWrapper(tracer=NoOpTracer(), span_name="messages.create")
    assert isinstance(wrapper._config, TraceConfig)
