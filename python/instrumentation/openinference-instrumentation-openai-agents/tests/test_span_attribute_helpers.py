from __future__ import annotations

import json
from secrets import token_hex
from typing import Any, Iterable, Mapping, Sequence, Union

import pytest
from agents.tracing.span_data import FunctionSpanData, GenerationSpanData, MCPListToolsSpanData
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionTool,
    Response,
    ResponseComputerToolCallOutputScreenshotParam,
    ResponseComputerToolCallParam,
    ResponseError,
    ResponseFileSearchToolCallParam,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseFunctionWebSearchParam,
    ResponseInputContentParam,
    ResponseInputItemParam,
    ResponseInputTextParam,
    ResponseOutputMessage,
    ResponseOutputMessageParam,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseOutputTextParam,
    ResponseReasoningItemParam,
    ResponseUsage,
    Tool,
)
from openai.types.responses.response_function_web_search_param import ActionSearch
from openai.types.responses.response_input_item_param import (
    ComputerCallOutput,
    FunctionCallOutput,
    ItemReference,
    Message,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from openinference.instrumentation.openai_agents._processor import (
    _get_attributes_from_chat_completions_input,
    _get_attributes_from_chat_completions_message_content,
    _get_attributes_from_chat_completions_message_content_item,
    _get_attributes_from_chat_completions_message_dicts,
    _get_attributes_from_chat_completions_output,
    _get_attributes_from_chat_completions_tool_call_dict,
    _get_attributes_from_chat_completions_usage,
    _get_attributes_from_function_call_output,
    _get_attributes_from_function_span_data,
    _get_attributes_from_function_tool_call,
    _get_attributes_from_generation_span_data,
    _get_attributes_from_input,
    _get_attributes_from_mcp_list_tool_span_data,
    _get_attributes_from_message,
    _get_attributes_from_message_content_list,
    _get_attributes_from_message_param,
    _get_attributes_from_response,
    _get_attributes_from_response_function_tool_call_param,
    _get_attributes_from_response_instruction,
    _get_attributes_from_response_output,
    _get_attributes_from_tools,
    _get_attributes_from_usage,
)


@pytest.mark.parametrize(
    "input_data,expected_attributes",
    [
        pytest.param(
            [
                Message(
                    role="user",
                    content=[ResponseInputTextParam(type="input_text", text="Hello")],
                    type="message",
                )
            ],
            {
                "llm.input_messages.1.message.contents.0.message_content.text": "Hello",
                "llm.input_messages.1.message.contents.0.message_content.type": "text",
                "llm.input_messages.1.message.role": "user",
            },
            id="simple_message",
        ),
        pytest.param(
            [
                ResponseFunctionToolCallParam(
                    type="function_call",
                    call_id="123",
                    name="test_func",
                    arguments="{}",
                )
            ],
            {
                "llm.input_messages.1.message.role": "assistant",
                "llm.input_messages.1.message.tool_calls.0.tool_call.id": "123",
                "llm.input_messages.1.message.tool_calls.0.tool_call.function.name": "test_func",
            },
            id="function_call",
        ),
        pytest.param(
            [
                Message(
                    role="user",
                    content=[
                        ResponseInputTextParam(
                            type="input_text",
                            text="Hello",
                        )
                    ],
                    type="message",
                ),
                ResponseOutputMessageParam(
                    id=token_hex(8),
                    status="completed",
                    role="assistant",
                    content=[
                        ResponseOutputTextParam(
                            type="output_text",
                            text="Hi",
                            annotations=[],
                        )
                    ],
                    type="message",
                ),
                ResponseFunctionToolCallParam(
                    type="function_call",
                    call_id="123",
                    name="test_func",
                    arguments="{}",
                ),
            ],
            {
                "llm.input_messages.1.message.contents.0.message_content.text": "Hello",
                "llm.input_messages.1.message.contents.0.message_content.type": "text",
                "llm.input_messages.1.message.role": "user",
                "llm.input_messages.2.message.contents.0.message_content.text": "Hi",
                "llm.input_messages.2.message.contents.0.message_content.type": "text",
                "llm.input_messages.2.message.role": "assistant",
                "llm.input_messages.3.message.tool_calls.0.tool_call.id": "123",
                "llm.input_messages.3.message.tool_calls.0.tool_call.function.name": "test_func",
                "llm.input_messages.3.message.role": "assistant",
            },
            id="multiple_messages",
        ),
        pytest.param(
            [
                Message(
                    role="developer",
                    content=[
                        ResponseInputTextParam(
                            type="input_text",
                            text="Debug info",
                        )
                    ],
                    type="message",
                )
            ],
            {
                "llm.input_messages.1.message.contents.0.message_content.text": "Debug info",
                "llm.input_messages.1.message.contents.0.message_content.type": "text",
                "llm.input_messages.1.message.role": "developer",
            },
            id="developer_message",
        ),
        pytest.param(
            [
                ResponseComputerToolCallParam(
                    type="computer_call",
                    id="comp-123",
                    call_id="call-123",
                    action={"type": "click", "x": 100, "y": 200, "button": "left"},
                    pending_safety_checks=[
                        {"id": "safety-1", "code": "SAFE", "message": "Action is safe"}
                    ],
                    status="in_progress",
                )
            ],
            {
                # TODO: Implement computer tool call attributes
            },
            id="computer_tool_call",
        ),
        pytest.param(
            [
                ResponseFileSearchToolCallParam(
                    type="file_search_call",
                    id="file-123",
                    queries=["test query"],
                    status="searching",
                )
            ],
            {
                # TODO: Implement file search tool call attributes
            },
            id="file_search_tool_call",
        ),
        pytest.param(
            [
                ResponseFunctionWebSearchParam(
                    type="web_search_call",
                    id="web-123",
                    status="searching",
                    action=ActionSearch(
                        type="search",
                        query="test query",
                    ),
                )
            ],
            {
                # TODO: Implement web search tool call attributes
            },
            id="web_search_tool_call",
        ),
        pytest.param(
            [
                ResponseReasoningItemParam(
                    type="reasoning",
                    id="reason-123",
                    summary=[{"type": "summary_text", "text": "Test reasoning"}],
                )
            ],
            {
                # TODO: Implement reasoning item attributes
            },
            id="reasoning_item",
        ),
        pytest.param(
            [
                ComputerCallOutput(
                    type="computer_call_output",
                    call_id="comp-123",
                    output=ResponseComputerToolCallOutputScreenshotParam(
                        type="computer_screenshot",
                        file_id="file-123",
                        image_url="https://example.com/screenshot.png",
                    ),
                    id="output-123",
                    status="completed",
                )
            ],
            {
                # TODO: Implement computer call output attributes
            },
            id="computer_call_output",
        ),
        pytest.param(
            [
                ItemReference(
                    type="item_reference",
                    id="ref-123",
                )
            ],
            {
                # TODO: Implement item reference attributes
            },
            id="item_reference",
        ),
    ],
)
def test_get_attributes_from_input(
    input_data: Iterable[ResponseInputItemParam],
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_input(input_data))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "message_param,expected_attributes",
    [
        pytest.param(
            EasyInputMessageParam(role="user", content="Hello", type="message"),
            {
                "message.role": "user",
                "message.content": "Hello",
            },
            id="simple_message",
        ),
        pytest.param(
            EasyInputMessageParam(
                role="assistant",
                content=[ResponseInputTextParam(type="input_text", text="Hi")],
                type="message",
            ),
            {
                "message.role": "assistant",
                "message.contents.0.message_content.type": "text",
                "message.contents.0.message_content.text": "Hi",
            },
            id="message_with_content_list",
        ),
        pytest.param(
            EasyInputMessageParam(role="developer", content="Debug info", type="message"),
            {
                "message.role": "developer",
                "message.content": "Debug info",
            },
            id="developer_message",
        ),
        pytest.param(
            EasyInputMessageParam(role="system", content="System message", type="message"),
            {
                "message.role": "system",
                "message.content": "System message",
            },
            id="system_message",
        ),
        pytest.param(
            EasyInputMessageParam(role="user", content=[], type="message"),
            {
                "message.role": "user",
            },
            id="empty_content_list",
        ),
        pytest.param(
            EasyInputMessageParam(
                role="user",
                content=[
                    ResponseInputTextParam(type="input_text", text="Hello"),
                    ResponseInputTextParam(type="input_text", text="World"),
                ],
                type="message",
            ),
            {
                "message.role": "user",
                "message.contents.0.message_content.type": "text",
                "message.contents.0.message_content.text": "Hello",
                "message.contents.1.message_content.type": "text",
                "message.contents.1.message_content.text": "World",
            },
            id="multiple_content_items",
        ),
    ],
)
def test_get_attributes_from_message_param(
    message_param: Union[EasyInputMessageParam, Message, ResponseOutputMessageParam],
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_message_param(message_param))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "tool_call_param,expected_attributes",
    [
        pytest.param(
            {
                "call_id": "123",
                "name": "test_func",
                "arguments": "{}",
            },
            {
                "tool_call.id": "123",
                "tool_call.function.name": "test_func",
            },
            id="empty_arguments",
        ),
        pytest.param(
            {
                "call_id": "123",
                "name": "test_func",
                "arguments": '{"arg": "value"}',
            },
            {
                "tool_call.id": "123",
                "tool_call.function.name": "test_func",
                "tool_call.function.arguments": '{"arg": "value"}',
            },
            id="with_arguments",
        ),
    ],
)
def test_get_attributes_from_response_function_tool_call_param(
    tool_call_param: ResponseFunctionToolCallParam,
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_response_function_tool_call_param(tool_call_param, ""))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "function_call_output,expected_attributes",
    [
        pytest.param(
            {
                "call_id": "123",
                "output": "result",
            },
            {
                "message.content": "result",
                "message.role": "tool",
                "message.tool_call_id": "123",
            },
            id="simple_output",
        ),
        pytest.param(
            {
                "call_id": "123",
                "output": "",
            },
            {
                "message.content": "",
                "message.role": "tool",
                "message.tool_call_id": "123",
            },
            id="empty_output",
        ),
        pytest.param(
            {
                "call_id": "123",
                "output": None,
            },
            {
                "message.content": None,
                "message.role": "tool",
                "message.tool_call_id": "123",
            },
            id="none_output",
        ),
    ],
)
def test_get_attributes_from_function_call_output(
    function_call_output: FunctionCallOutput,
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_function_call_output(function_call_output, ""))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "generation_span_data,expected_attributes",
    [
        pytest.param(
            GenerationSpanData(
                model="gpt-4",
                model_config={"temperature": 0.7},
                input=[{"role": "user", "content": "Hello"}],
                output=[{"role": "assistant", "content": "Hi"}],
                usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            ),
            {
                "input.mime_type": "application/json",
                "input.value": '[{"role": "user", "content": "Hello"}]',
                "llm.input_messages.0.message.content": "Hello",
                "llm.input_messages.0.message.role": "user",
                "llm.invocation_parameters": '{"temperature": 0.7}',
                "llm.model_name": "gpt-4",
                "llm.output_messages.0.message.content": "Hi",
                "llm.output_messages.0.message.role": "assistant",
                "llm.token_count.completion": 5,
                "llm.token_count.prompt": 10,
                "output.mime_type": "application/json",
                "output.value": '[{"role": "assistant", "content": "Hi"}]',
            },
            id="complete_generation",
        ),
        pytest.param(
            GenerationSpanData(
                model="gpt-4",
                model_config=None,
                input=None,
                output=None,
                usage=None,
            ),
            {
                "llm.model_name": "gpt-4",
            },
            id="minimal_generation",
        ),
        pytest.param(
            GenerationSpanData(
                model="gpt-4",
                model_config={"temperature": 0.7, "base_url": "https://api.openai.com/v1"},
                input=[{"role": "user", "content": "Hello"}],
                output=[{"role": "assistant", "content": "Hi"}],
                usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            ),
            {
                "input.mime_type": "application/json",
                "input.value": '[{"role": "user", "content": "Hello"}]',
                "llm.input_messages.0.message.content": "Hello",
                "llm.input_messages.0.message.role": "user",
                "llm.invocation_parameters": '{"temperature": 0.7, "base_url": "https://api.openai.com/v1"}',
                "llm.model_name": "gpt-4",
                "llm.provider": "openai",
                "llm.output_messages.0.message.content": "Hi",
                "llm.output_messages.0.message.role": "assistant",
                "llm.token_count.completion": 5,
                "llm.token_count.prompt": 10,
                "output.mime_type": "application/json",
                "output.value": '[{"role": "assistant", "content": "Hi"}]',
            },
            id="generation_with_provider",
        ),
        pytest.param(
            GenerationSpanData(
                model="gpt-4",
                model_config={"temperature": 0.7, "base_url": "https://other-api.com"},
                input=[{"role": "user", "content": "Hello"}],
                output=[{"role": "assistant", "content": "Hi"}],
                usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            ),
            {
                "input.mime_type": "application/json",
                "input.value": '[{"role": "user", "content": "Hello"}]',
                "llm.input_messages.0.message.content": "Hello",
                "llm.input_messages.0.message.role": "user",
                "llm.invocation_parameters": '{"temperature": 0.7, "base_url": "https://other-api.com"}',
                "llm.model_name": "gpt-4",
                "llm.output_messages.0.message.content": "Hi",
                "llm.output_messages.0.message.role": "assistant",
                "llm.token_count.completion": 5,
                "llm.token_count.prompt": 10,
                "output.mime_type": "application/json",
                "output.value": '[{"role": "assistant", "content": "Hi"}]',
            },
            id="generation_with_other_provider",
        ),
    ],
)
def test_get_attributes_from_generation_span_data(
    generation_span_data: GenerationSpanData,
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_generation_span_data(generation_span_data))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "mcp_list_tool_span_data,expected_attributes",
    [
        pytest.param(
            MCPListToolsSpanData(server="test-server", result=["tool1", "tool2"]),
            {
                "output.value": '["tool1", "tool2"]',
                "output.mime_type": "application/json",
            },
            id="complete_tools_list",
        ),
        pytest.param(
            MCPListToolsSpanData(server="test-server", result=[]),
            {
                "output.value": "[]",
                "output.mime_type": "application/json",
            },
            id="empty_tools_list",
        ),
        pytest.param(
            MCPListToolsSpanData(server="test-server", result=None),
            {
                "output.value": "null",
                "output.mime_type": "application/json",
            },
            id="none_tools_list",
        ),
    ],
)
def test_get_attributes_from_mcp_list_tool_span_data(
    mcp_list_tool_span_data: MCPListToolsSpanData,
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_mcp_list_tool_span_data(mcp_list_tool_span_data))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "chat_completions_input,expected_attributes",
    [
        pytest.param(
            [{"role": "user", "content": "Hello"}],
            {
                "input.value": '[{"role": "user", "content": "Hello"}]',
                "input.mime_type": "application/json",
                "llm.input_messages.0.message.role": "user",
                "llm.input_messages.0.message.content": "Hello",
            },
            id="simple_input",
        ),
        pytest.param(
            [],
            {},
            id="empty_input",
        ),
        pytest.param(
            None,
            {},
            id="none_input",
        ),
        pytest.param(
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
            {
                "input.value": json.dumps(
                    [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi"},
                    ]
                ),
                "input.mime_type": "application/json",
                "llm.input_messages.0.message.role": "user",
                "llm.input_messages.0.message.content": "Hello",
                "llm.input_messages.1.message.role": "assistant",
                "llm.input_messages.1.message.content": "Hi",
            },
            id="multiple_messages",
        ),
    ],
)
def test_get_attributes_from_chat_completions_input(
    chat_completions_input: Sequence[dict[str, Any]],
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_chat_completions_input(chat_completions_input))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "chat_completions_output,expected_attributes",
    [
        pytest.param(
            [{"role": "assistant", "content": "Hi"}],
            {
                "output.value": json.dumps(
                    [
                        {"role": "assistant", "content": "Hi"},
                    ]
                ),
                "output.mime_type": "application/json",
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.content": "Hi",
            },
            id="simple_output",
        ),
        pytest.param(
            [],
            {},
            id="empty_output",
        ),
        pytest.param(
            None,
            {},
            id="none_output",
        ),
        pytest.param(
            [{"role": "assistant", "content": "Hi"}, {"role": "user", "content": "Thanks"}],
            {
                "output.value": json.dumps(
                    [
                        {"role": "assistant", "content": "Hi"},
                        {"role": "user", "content": "Thanks"},
                    ]
                ),
                "output.mime_type": "application/json",
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.content": "Hi",
                "llm.output_messages.1.message.role": "user",
                "llm.output_messages.1.message.content": "Thanks",
            },
            id="multiple_messages",
        ),
    ],
)
def test_get_attributes_from_chat_completions_output(
    chat_completions_output: Sequence[dict[str, Any]],
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_chat_completions_output(chat_completions_output))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "message_dicts,expected_attributes",
    [
        pytest.param(
            [{"role": "user", "content": "Hello"}],
            {
                "llm.input_messages.0.message.role": "user",
                "llm.input_messages.0.message.content": "Hello",
            },
            id="simple_message",
        ),
        pytest.param(
            [
                {
                    "role": "assistant",
                    "content": "Hi",
                    "tool_calls": [{"id": "123", "function": {"name": "test_func"}}],
                }
            ],
            {
                "llm.input_messages.0.message.role": "assistant",
                "llm.input_messages.0.message.content": "Hi",
                "llm.input_messages.0.message.tool_calls.0.tool_call.id": "123",
                "llm.input_messages.0.message.tool_calls.0.tool_call.function.name": "test_func",
            },
            id="message_with_tool_calls",
        ),
        pytest.param(
            [
                {
                    "role": "assistant",
                    "content": "Hi",
                    "tool_calls": [
                        {"id": "123", "function": {"name": "test_func1"}},
                        {"id": "456", "function": {"name": "test_func2"}},
                    ],
                }
            ],
            {
                "llm.input_messages.0.message.role": "assistant",
                "llm.input_messages.0.message.content": "Hi",
                "llm.input_messages.0.message.tool_calls.0.tool_call.id": "123",
                "llm.input_messages.0.message.tool_calls.0.tool_call.function.name": "test_func1",
                "llm.input_messages.0.message.tool_calls.1.tool_call.id": "456",
                "llm.input_messages.0.message.tool_calls.1.tool_call.function.name": "test_func2",
            },
            id="message_with_multiple_tool_calls",
        ),
        pytest.param(
            [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "Hi",
                    "tool_calls": [
                        {
                            "id": "123",
                            "function": {"name": "test_func"},
                        },
                    ],
                },
            ],
            {
                "llm.input_messages.0.message.role": "user",
                "llm.input_messages.0.message.content": "Hello",
                "llm.input_messages.1.message.role": "assistant",
                "llm.input_messages.1.message.content": "Hi",
                "llm.input_messages.1.message.tool_calls.0.tool_call.id": "123",
                "llm.input_messages.1.message.tool_calls.0.tool_call.function.name": "test_func",
            },
            id="multiple_messages_with_tool_calls",
        ),
        pytest.param(
            [{"role": "user"}],
            {
                "llm.input_messages.0.message.role": "user",
            },
            id="message_without_content",
        ),
        pytest.param(
            [{"content": "Hello"}],
            {
                "llm.input_messages.0.message.content": "Hello",
            },
            id="message_without_role",
        ),
        pytest.param(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "123", "function": {"name": "test_func"}}],
                }
            ],
            {
                "llm.input_messages.0.message.role": "assistant",
                "llm.input_messages.0.message.tool_calls.0.tool_call.id": "123",
                "llm.input_messages.0.message.tool_calls.0.tool_call.function.name": "test_func",
            },
            id="message_without_content_but_with_tool_calls",
        ),
    ],
)
def test_get_attributes_from_chat_completions_message_dicts(
    message_dicts: Sequence[dict[str, Any]],
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(
        _get_attributes_from_chat_completions_message_dicts(message_dicts, "llm.input_messages.")
    )
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "message_content,expected_attributes",
    [
        pytest.param(
            "Hello",
            {"llm.input_messages.0.message.content": "Hello"},
            id="simple_text",
        ),
        pytest.param(
            [{"type": "text", "text": "Hi"}],
            {
                "llm.input_messages.0.message.contents.0.message_content.type": "text",
                "llm.input_messages.0.message.contents.0.message_content.text": "Hi",
            },
            id="content_list",
        ),
        pytest.param(
            None,
            {},
            id="none_content",
        ),
        pytest.param(
            [],
            {},
            id="empty_content",
        ),
        pytest.param(
            [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}],
            {
                "llm.input_messages.0.message.contents.0.message_content.type": "text",
                "llm.input_messages.0.message.contents.0.message_content.text": "Hello",
                "llm.input_messages.0.message.contents.1.message_content.type": "text",
                "llm.input_messages.0.message.contents.1.message_content.text": "World",
            },
            id="multiple_content_items",
        ),
    ],
)
def test_get_attributes_from_chat_completions_message_content(
    message_content: Union[str, Iterable[Mapping[str, Any]]],
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(
        _get_attributes_from_chat_completions_message_content(
            message_content,
            "llm.input_messages.0.",
        )
    )
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "content_item,expected_attributes",
    [
        pytest.param(
            {"type": "text", "text": "Hello"},
            {
                "message_content.type": "text",
                "message_content.text": "Hello",
            },
            id="text_content",
        ),
        pytest.param(
            {"type": "text", "text": None},
            {},
            id="empty_text",
        ),
        pytest.param(
            {"type": "text"},
            {},
            id="no_text",
        ),
        pytest.param(
            {"type": "other"},
            {},
            id="other_type",
        ),
    ],
)
def test_get_attributes_from_chat_completions_message_content_item(
    content_item: Mapping[str, Any],
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_chat_completions_message_content_item(content_item, ""))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "tool_call_dict,expected_attributes",
    [
        pytest.param(
            {
                "id": "123",
                "function": {"name": "test_func", "arguments": '{"arg": "value"}'},
            },
            {
                "tool_call.id": "123",
                "tool_call.function.name": "test_func",
                "tool_call.function.arguments": '{"arg": "value"}',
            },
            id="complete_tool_call",
        ),
        pytest.param(
            {
                "id": "123",
                "function": {"name": "test_func", "arguments": "{}"},
            },
            {
                "tool_call.id": "123",
                "tool_call.function.name": "test_func",
            },
            id="empty_arguments",
        ),
        pytest.param(
            {
                "id": "123",
                "function": {"name": "test_func"},
            },
            {
                "tool_call.id": "123",
                "tool_call.function.name": "test_func",
            },
            id="no_arguments",
        ),
        pytest.param(
            {
                "id": "123",
            },
            {
                "tool_call.id": "123",
            },
            id="no_function",
        ),
        pytest.param(
            {
                "function": {"name": "test_func"},
            },
            {
                "tool_call.function.name": "test_func",
            },
            id="no_id",
        ),
    ],
)
def test_get_attributes_from_chat_completions_tool_call_dict(
    tool_call_dict: Mapping[str, Any],
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_chat_completions_tool_call_dict(tool_call_dict, ""))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "usage_dict,expected_attributes",
    [
        pytest.param(
            {"input_tokens": 10, "output_tokens": 5},
            {
                "llm.token_count.prompt": 10,
                "llm.token_count.completion": 5,
            },
            id="complete_usage",
        ),
        pytest.param(
            None,
            {},
            id="none_usage",
        ),
        pytest.param(
            {},
            {},
            id="empty_usage",
        ),
        pytest.param(
            {"input_tokens": 10},
            {
                "llm.token_count.prompt": 10,
            },
            id="only_input_tokens",
        ),
        pytest.param(
            {"output_tokens": 5},
            {
                "llm.token_count.completion": 5,
            },
            id="only_output_tokens",
        ),
        pytest.param(
            {"input_tokens": 0, "output_tokens": 0},
            {},
            id="zero_tokens",
        ),
    ],
)
def test_get_attributes_from_chat_completions_usage(
    usage_dict: Mapping[str, Any],
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_chat_completions_usage(usage_dict))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "function_span_data,expected_attributes",
    [
        pytest.param(
            FunctionSpanData(
                name="test_func",
                input=json.dumps({"k": "v"}),
                output=json.dumps({"result": "success"}),
                mcp_data={"key": "value"},
            ),
            {
                "tool.name": "test_func",
                "input.value": '{"k": "v"}',
                "input.mime_type": "application/json",
                "output.value": '{"result": "success"}',
                "output.mime_type": "application/json",
            },
            id="complete_function",
        ),
        pytest.param(
            FunctionSpanData(
                name="test_func",
                input=None,
                output=None,
                mcp_data=None,
            ),
            {
                "tool.name": "test_func",
            },
            id="minimal_function",
        ),
        pytest.param(
            FunctionSpanData(
                name="test_func",
                input=json.dumps({"complex": {"nested": "data"}}),
                output=json.dumps({"result": "success"}),
                mcp_data={"metadata": "value"},
            ),
            {
                "tool.name": "test_func",
                "input.value": '{"complex": {"nested": "data"}}',
                "input.mime_type": "application/json",
                "output.value": '{"result": "success"}',
                "output.mime_type": "application/json",
            },
            id="complex_json_data",
        ),
        pytest.param(
            FunctionSpanData(
                name="test_func",
                input=json.dumps({"k": "v"}),
                output="",
                mcp_data=None,
            ),
            {
                "tool.name": "test_func",
                "input.value": '{"k": "v"}',
                "input.mime_type": "application/json",
                "output.value": "",
            },
            id="empty_string_output",
        ),
    ],
)
def test_get_attributes_from_function_span_data(
    function_span_data: FunctionSpanData,
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_function_span_data(function_span_data))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "message_content_list,expected_attributes",
    [
        pytest.param(
            [{"type": "input_text", "text": "Hello"}],
            {
                "message.contents.0.message_content.type": "text",
                "message.contents.0.message_content.text": "Hello",
            },
            id="input_text",
        ),
        pytest.param(
            [{"type": "output_text", "text": "Hi"}],
            {
                "message.contents.0.message_content.type": "text",
                "message.contents.0.message_content.text": "Hi",
            },
            id="output_text",
        ),
        pytest.param(
            [],
            {},
            id="empty_content_list",
        ),
        pytest.param(
            [
                {"type": "input_text", "text": "Hello"},
                {"type": "output_text", "text": "Hi"},
            ],
            {
                "message.contents.0.message_content.type": "text",
                "message.contents.0.message_content.text": "Hello",
                "message.contents.1.message_content.type": "text",
                "message.contents.1.message_content.text": "Hi",
            },
            id="multiple_content_items",
        ),
        pytest.param(
            [{"type": "refusal", "refusal": "I cannot help with that"}],
            {
                "message.contents.0.message_content.type": "text",
                "message.contents.0.message_content.text": "I cannot help with that",
            },
            id="refusal_content",
        ),
        pytest.param(
            [
                {"type": "input_text", "text": "Hello"},
                {"type": "refusal", "refusal": "I cannot help with that"},
            ],
            {
                "message.contents.0.message_content.type": "text",
                "message.contents.0.message_content.text": "Hello",
                "message.contents.1.message_content.type": "text",
                "message.contents.1.message_content.text": "I cannot help with that",
            },
            id="mixed_content_types",
        ),
    ],
)
def test_get_attributes_from_message_content_list(
    message_content_list: Sequence[ResponseInputContentParam],
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_message_content_list(message_content_list))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "response,expected_attributes",
    [
        pytest.param(
            Response(
                id="test-id",
                created_at=1234567890.0,
                model="gpt-4",
                object="response",
                output=[
                    ResponseOutputMessage(
                        id=token_hex(8),
                        type="message",
                        status="completed",
                        role="assistant",
                        content=[
                            ResponseOutputText(
                                text="Hi",
                                type="output_text",
                                annotations=[],
                            )
                        ],
                    )
                ],
                parallel_tool_calls=True,
                tool_choice="auto",
                tools=[
                    FunctionTool(
                        type="function",
                        name="test_func",
                        description="test",
                        parameters={},
                        strict=True,
                    )
                ],
                usage=ResponseUsage(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    input_tokens_details=InputTokensDetails(
                        cached_tokens=0,
                    ),
                    output_tokens_details=OutputTokensDetails(
                        reasoning_tokens=0,
                    ),
                ),
                instructions="Be helpful",
                temperature=0.7,
                top_p=0.9,
                max_output_tokens=100,
                previous_response_id="prev-id",
                status="completed",
                truncation="auto",
                user="test-user",
            ),
            {
                "llm.input_messages.0.message.content": "Be helpful",
                "llm.input_messages.0.message.role": "system",
                "llm.invocation_parameters": json.dumps(
                    {
                        "id": "test-id",
                        "created_at": 1234567890.0,
                        "instructions": "Be helpful",
                        "model": "gpt-4",
                        "parallel_tool_calls": True,
                        "temperature": 0.7,
                        "tool_choice": "auto",
                        "top_p": 0.9,
                        "max_output_tokens": 100,
                        "previous_response_id": "prev-id",
                        "truncation": "auto",
                        "user": "test-user",
                    }
                ),
                "llm.model_name": "gpt-4",
                "llm.output_messages.0.message.contents.0.message_content.text": "Hi",
                "llm.output_messages.0.message.contents.0.message_content.type": "text",
                "llm.output_messages.0.message.role": "assistant",
                "llm.token_count.completion": 5,
                "llm.token_count.completion_details.reasoning": 0,
                "llm.token_count.prompt": 10,
                "llm.token_count.prompt_details.cache_read": 0,
                "llm.token_count.total": 15,
                "llm.tools.0.tool.json_schema": json.dumps(
                    {
                        "type": "function",
                        "function": {
                            "name": "test_func",
                            "description": "test",
                            "parameters": {},
                            "strict": True,
                        },
                    }
                ),
            },
            id="complete_response",
        ),
        pytest.param(
            Response(
                id="incomplete-id",
                created_at=1234567890.0,
                model="gpt-4",
                object="response",
                output=[],
                parallel_tool_calls=True,
                tool_choice="auto",
                tools=[],
                status="incomplete",
            ),
            {
                "llm.invocation_parameters": json.dumps(
                    {
                        "id": "incomplete-id",
                        "created_at": 1234567890.0,
                        "model": "gpt-4",
                        "parallel_tool_calls": True,
                        "tool_choice": "auto",
                    }
                ),
                "llm.model_name": "gpt-4",
            },
            id="incomplete_response",
        ),
        pytest.param(
            Response(
                id="minimal-id",
                created_at=1234567890.0,
                model="gpt-4",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
                status="completed",
            ),
            {
                "llm.invocation_parameters": json.dumps(
                    {
                        "id": "minimal-id",
                        "created_at": 1234567890.0,
                        "model": "gpt-4",
                        "parallel_tool_calls": False,
                        "tool_choice": "none",
                    }
                ),
                "llm.model_name": "gpt-4",
            },
            id="minimal_response",
        ),
        pytest.param(
            Response(
                id="error-id",
                created_at=1234567890.0,
                model="gpt-4",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
                status="failed",
                error=ResponseError(
                    code="rate_limit_exceeded",
                    message="Rate limit exceeded",
                ),
            ),
            {
                "llm.invocation_parameters": json.dumps(
                    {
                        "id": "error-id",
                        "created_at": 1234567890.0,
                        "model": "gpt-4",
                        "parallel_tool_calls": False,
                        "tool_choice": "none",
                    }
                ),
                "llm.model_name": "gpt-4",
            },
            id="error_response",
        ),
        pytest.param(
            Response(
                id="complex-id",
                created_at=1234567890.0,
                model="gpt-4",
                object="response",
                output=[
                    ResponseOutputMessage(
                        id=token_hex(8),
                        type="message",
                        status="completed",
                        role="assistant",
                        content=[
                            ResponseOutputText(
                                text="Hi",
                                type="output_text",
                                annotations=[],
                            ),
                            ResponseOutputRefusal(
                                type="refusal",
                                refusal="I cannot help with that",
                            ),
                        ],
                    ),
                    ResponseFunctionToolCall(
                        type="function_call",
                        call_id="123",
                        name="test_func",
                        arguments='{"arg": "value"}',
                    ),
                ],
                parallel_tool_calls=True,
                tool_choice="auto",
                tools=[
                    FunctionTool(
                        type="function",
                        name="test_func1",
                        description="test1",
                        parameters={"type": "object", "properties": {}},
                        strict=True,
                    ),
                    FunctionTool(
                        type="function",
                        name="test_func2",
                        description="test2",
                        parameters={"type": "object", "properties": {}},
                        strict=True,
                    ),
                ],
                usage=ResponseUsage(
                    input_tokens=1000,
                    output_tokens=500,
                    total_tokens=1500,
                    input_tokens_details=InputTokensDetails(
                        cached_tokens=100,
                    ),
                    output_tokens_details=OutputTokensDetails(
                        reasoning_tokens=50,
                    ),
                ),
                instructions="Be helpful\nAnd friendly",
                temperature=0.7,
                top_p=0.9,
                max_output_tokens=100,
                previous_response_id="prev-id",
                status="completed",
                truncation="auto",
                user="test-user",
            ),
            {
                "llm.input_messages.0.message.content": "Be helpful\nAnd friendly",
                "llm.input_messages.0.message.role": "system",
                "llm.invocation_parameters": json.dumps(
                    {
                        "id": "complex-id",
                        "created_at": 1234567890.0,
                        "instructions": "Be helpful\nAnd friendly",
                        "model": "gpt-4",
                        "parallel_tool_calls": True,
                        "temperature": 0.7,
                        "tool_choice": "auto",
                        "top_p": 0.9,
                        "max_output_tokens": 100,
                        "previous_response_id": "prev-id",
                        "truncation": "auto",
                        "user": "test-user",
                    }
                ),
                "llm.model_name": "gpt-4",
                "llm.output_messages.0.message.contents.0.message_content.text": "Hi",
                "llm.output_messages.0.message.contents.0.message_content.type": "text",
                "llm.output_messages.0.message.contents.1.message_content.text": "I cannot help with that",  # noqa: E501
                "llm.output_messages.0.message.contents.1.message_content.type": "text",
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.1.message.role": "assistant",
                "llm.output_messages.1.message.tool_calls.0.tool_call.id": "123",
                "llm.output_messages.1.message.tool_calls.0.tool_call.function.name": "test_func",
                "llm.output_messages.1.message.tool_calls.0.tool_call.function.arguments": '{"arg": "value"}',  # noqa: E501
                "llm.token_count.completion": 500,
                "llm.token_count.completion_details.reasoning": 50,
                "llm.token_count.prompt": 1000,
                "llm.token_count.prompt_details.cache_read": 100,
                "llm.token_count.total": 1500,
                "llm.tools.0.tool.json_schema": json.dumps(
                    {
                        "type": "function",
                        "function": {
                            "name": "test_func1",
                            "description": "test1",
                            "parameters": {"type": "object", "properties": {}},
                            "strict": True,
                        },
                    }
                ),
                "llm.tools.1.tool.json_schema": json.dumps(
                    {
                        "type": "function",
                        "function": {
                            "name": "test_func2",
                            "description": "test2",
                            "parameters": {"type": "object", "properties": {}},
                            "strict": True,
                        },
                    }
                ),
            },
            id="complex_response",
        ),
    ],
)
def test_get_attributes_from_response(
    response: Response,
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_response(response))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "tools,expected_attributes",
    [
        pytest.param(
            [
                FunctionTool(
                    name="test_func",
                    description="test",
                    parameters={},
                    strict=True,
                    type="function",
                )
            ],
            {
                "llm.tools.0.tool.json_schema": json.dumps(
                    {
                        "type": "function",
                        "function": {
                            "name": "test_func",
                            "description": "test",
                            "parameters": {},
                            "strict": True,
                        },
                    }
                ),
            },
            id="function_tool",
        ),
        pytest.param(
            [
                FunctionTool(
                    name="test_func",
                    parameters={},
                    strict=True,
                    type="function",
                )
            ],
            {
                "llm.tools.0.tool.json_schema": json.dumps(
                    {
                        "type": "function",
                        "function": {
                            "name": "test_func",
                            "description": None,
                            "parameters": {},
                            "strict": True,
                        },
                    }
                ),
            },
            id="function_tool_no_description",
        ),
        pytest.param(
            [],
            {},
            id="empty_tools",
        ),
        pytest.param(
            None,
            {},
            id="none_tools",
        ),
        pytest.param(
            [
                FunctionTool(
                    name="test_func1",
                    description="test1",
                    parameters={
                        "type": "object",
                        "properties": {},
                    },
                    strict=True,
                    type="function",
                ),
                FunctionTool(
                    name="test_func2",
                    description="test2",
                    parameters={
                        "type": "object",
                        "properties": {},
                    },
                    strict=True,
                    type="function",
                ),
            ],
            {
                "llm.tools.0.tool.json_schema": json.dumps(
                    {
                        "type": "function",
                        "function": {
                            "name": "test_func1",
                            "description": "test1",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                            "strict": True,
                        },
                    }
                ),
                "llm.tools.1.tool.json_schema": json.dumps(
                    {
                        "type": "function",
                        "function": {
                            "name": "test_func2",
                            "description": "test2",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                            "strict": True,
                        },
                    }
                ),
            },
            id="multiple_tools",
        ),
        pytest.param(
            [
                FunctionTool(
                    name="test_func",
                    description="test",
                    parameters={
                        "type": "object",
                        "properties": {
                            "arg1": {
                                "type": "string",
                                "description": "First argument",
                            },
                            "arg2": {
                                "type": "number",
                                "description": "Second argument",
                            },
                        },
                        "required": ["arg1"],
                    },
                    strict=True,
                    type="function",
                )
            ],
            {
                "llm.tools.0.tool.json_schema": json.dumps(
                    {
                        "type": "function",
                        "function": {
                            "name": "test_func",
                            "description": "test",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "arg1": {
                                        "type": "string",
                                        "description": "First argument",
                                    },
                                    "arg2": {
                                        "type": "number",
                                        "description": "Second argument",
                                    },
                                },
                                "required": ["arg1"],
                            },
                            "strict": True,
                        },
                    }
                ),
            },
            id="function_tool_with_complex_parameters",
        ),
        pytest.param(
            [
                FunctionTool(
                    name="test_func",
                    description="test",
                    parameters={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                    strict=False,
                    type="function",
                )
            ],
            {
                "llm.tools.0.tool.json_schema": json.dumps(
                    {
                        "type": "function",
                        "function": {
                            "name": "test_func",
                            "description": "test",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": False,
                            },
                            "strict": False,
                        },
                    }
                ),
            },
            id="function_tool_with_additional_properties",
        ),
    ],
)
def test_get_attributes_from_tools(
    tools: Sequence[Tool],
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_tools(tools))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "response_output,expected_attributes",
    [
        pytest.param(
            [
                ResponseOutputMessage(
                    id="msg-123",
                    role="assistant",
                    content=[
                        ResponseOutputText(
                            type="output_text",
                            text="Hi",
                            annotations=[],
                        )
                    ],
                    status="completed",
                    type="message",
                )
            ],
            {
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.contents.0.message_content.type": "text",
                "llm.output_messages.0.message.contents.0.message_content.text": "Hi",
            },
            id="message_output",
        ),
        pytest.param(
            [
                ResponseFunctionToolCall(
                    type="function_call",
                    call_id="123",
                    name="test_func",
                    arguments="{}",
                )
            ],
            {
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.tool_calls.0.tool_call.id": "123",
                "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "test_func",
            },
            id="function_call_output",
        ),
        pytest.param(
            [
                ResponseOutputMessage(
                    id="msg-123",
                    role="assistant",
                    content=[
                        ResponseOutputText(
                            type="output_text",
                            text="Hi",
                            annotations=[],
                        )
                    ],
                    status="completed",
                    type="message",
                ),
                ResponseFunctionToolCall(
                    type="function_call",
                    call_id="123",
                    name="test_func",
                    arguments="{}",
                ),
            ],
            {
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.contents.0.message_content.type": "text",
                "llm.output_messages.0.message.contents.0.message_content.text": "Hi",
                "llm.output_messages.1.message.role": "assistant",
                "llm.output_messages.1.message.tool_calls.0.tool_call.id": "123",
                "llm.output_messages.1.message.tool_calls.0.tool_call.function.name": "test_func",
            },
            id="multiple_outputs",
        ),
        pytest.param(
            [
                ResponseFunctionToolCall(
                    type="function_call",
                    call_id="123",
                    name="test_func1",
                    arguments="{}",
                ),
                ResponseFunctionToolCall(
                    type="function_call",
                    call_id="456",
                    name="test_func2",
                    arguments="{}",
                ),
            ],
            {
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.tool_calls.0.tool_call.id": "123",
                "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "test_func1",
                "llm.output_messages.0.message.tool_calls.1.tool_call.id": "456",
                "llm.output_messages.0.message.tool_calls.1.tool_call.function.name": "test_func2",
            },
            id="multiple_function_calls",
        ),
        pytest.param(
            [
                ResponseOutputMessage(
                    id="msg-123",
                    role="assistant",
                    content=[
                        ResponseOutputText(
                            type="output_text",
                            text="Hi",
                            annotations=[],
                        )
                    ],
                    status="completed",
                    type="message",
                ),
                ResponseOutputMessage(
                    id="msg-124",
                    role="assistant",
                    content=[
                        ResponseOutputText(
                            type="output_text",
                            text="World",
                            annotations=[],
                        )
                    ],
                    status="completed",
                    type="message",
                ),
            ],
            {
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.contents.0.message_content.type": "text",
                "llm.output_messages.0.message.contents.0.message_content.text": "Hi",
                "llm.output_messages.1.message.role": "assistant",
                "llm.output_messages.1.message.contents.0.message_content.type": "text",
                "llm.output_messages.1.message.contents.0.message_content.text": "World",
            },
            id="multiple_messages",
        ),
        pytest.param(
            [],
            {},
            id="empty_output",
        ),
    ],
)
def test_get_attributes_from_response_output(
    response_output: Sequence[Any],
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_response_output(response_output))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "instructions,expected_attributes",
    [
        pytest.param(
            "Be helpful",
            {
                "llm.input_messages.0.message.role": "system",
                "llm.input_messages.0.message.content": "Be helpful",
            },
            id="with_instructions",
        ),
        pytest.param(
            None,
            {},
            id="no_instructions",
        ),
        pytest.param(
            "",
            {},
            id="empty_instructions",
        ),
        pytest.param(
            "Be helpful\nAnd friendly",
            {
                "llm.input_messages.0.message.role": "system",
                "llm.input_messages.0.message.content": "Be helpful\nAnd friendly",
            },
            id="multiline_instructions",
        ),
    ],
)
def test_get_attributes_from_response_instruction(
    instructions: str | None,
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_response_instruction(instructions))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "function_tool_call,expected_attributes",
    [
        pytest.param(
            ResponseFunctionToolCall(
                type="function_call",
                call_id="123",
                name="test_func",
                arguments="{}",
            ),
            {
                "tool_call.id": "123",
                "tool_call.function.name": "test_func",
            },
            id="empty_arguments",
        ),
        pytest.param(
            ResponseFunctionToolCall(
                type="function_call",
                call_id="123",
                name="test_func",
                arguments='{"arg": "value"}',
            ),
            {
                "tool_call.id": "123",
                "tool_call.function.name": "test_func",
                "tool_call.function.arguments": '{"arg": "value"}',
            },
            id="with_arguments",
        ),
    ],
)
def test_get_attributes_from_function_tool_call(
    function_tool_call: ResponseFunctionToolCall,
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_function_tool_call(function_tool_call, ""))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "message,expected_attributes",
    [
        pytest.param(
            ResponseOutputMessage(
                id="msg-123",
                role="assistant",
                content=[
                    ResponseOutputText(
                        text="Hi",
                        type="output_text",
                        annotations=[],
                    )
                ],
                status="completed",
                type="message",
            ),
            {
                "message.role": "assistant",
                "message.contents.0.message_content.type": "text",
                "message.contents.0.message_content.text": "Hi",
            },
            id="text_message",
        ),
        pytest.param(
            ResponseOutputMessage(
                id="msg-124",
                role="assistant",
                content=[
                    ResponseOutputRefusal(
                        type="refusal",
                        refusal="I cannot help with that",
                    )
                ],
                status="completed",
                type="message",
            ),
            {
                "message.role": "assistant",
                "message.contents.0.message_content.type": "text",
                "message.contents.0.message_content.text": "I cannot help with that",
            },
            id="refusal_message",
        ),
        pytest.param(
            ResponseOutputMessage(
                id="msg-125",
                role="assistant",
                content=[
                    ResponseOutputText(
                        text="Hi",
                        type="output_text",
                        annotations=[],
                    ),
                    ResponseOutputRefusal(
                        type="refusal",
                        refusal="I cannot help with that",
                    ),
                ],
                status="in_progress",
                type="message",
            ),
            {
                "message.role": "assistant",
                "message.contents.0.message_content.type": "text",
                "message.contents.0.message_content.text": "Hi",
                "message.contents.1.message_content.type": "text",
                "message.contents.1.message_content.text": "I cannot help with that",
            },
            id="mixed_content_message",
        ),
        pytest.param(
            ResponseOutputMessage(
                id="msg-126",
                role="assistant",
                content=[],
                status="incomplete",
                type="message",
            ),
            {
                "message.role": "assistant",
            },
            id="empty_content_message",
        ),
        pytest.param(
            ResponseOutputMessage(
                id="msg-130",
                role="assistant",
                content=[
                    ResponseOutputText(
                        text="Hello",
                        type="output_text",
                        annotations=[],
                    ),
                    ResponseOutputText(
                        text="World",
                        type="output_text",
                        annotations=[],
                    ),
                ],
                status="completed",
                type="message",
            ),
            {
                "message.role": "assistant",
                "message.contents.0.message_content.type": "text",
                "message.contents.0.message_content.text": "Hello",
                "message.contents.1.message_content.type": "text",
                "message.contents.1.message_content.text": "World",
            },
            id="multiple_text_messages",
        ),
    ],
)
def test_get_attributes_from_message(
    message: ResponseOutputMessage,
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_message(message))
    assert attributes == expected_attributes


@pytest.mark.parametrize(
    "usage,expected_attributes",
    [
        pytest.param(
            ResponseUsage(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                input_tokens_details=InputTokensDetails(
                    cached_tokens=0,
                ),
                output_tokens_details=OutputTokensDetails(
                    reasoning_tokens=0,
                ),
            ),
            {
                "llm.token_count.prompt": 10,
                "llm.token_count.completion_details.reasoning": 0,
                "llm.token_count.completion": 5,
                "llm.token_count.prompt_details.cache_read": 0,
                "llm.token_count.total": 15,
            },
            id="complete_usage",
        ),
        pytest.param(
            None,
            {},
            id="no_usage",
        ),
        pytest.param(
            ResponseUsage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                input_tokens_details=InputTokensDetails(
                    cached_tokens=0,
                ),
                output_tokens_details=OutputTokensDetails(
                    reasoning_tokens=0,
                ),
            ),
            {
                "llm.token_count.prompt": 0,
                "llm.token_count.completion_details.reasoning": 0,
                "llm.token_count.completion": 0,
                "llm.token_count.prompt_details.cache_read": 0,
                "llm.token_count.total": 0,
            },
            id="zero_tokens",
        ),
        pytest.param(
            ResponseUsage(
                input_tokens=1000,
                output_tokens=500,
                total_tokens=1500,
                input_tokens_details=InputTokensDetails(
                    cached_tokens=100,
                ),
                output_tokens_details=OutputTokensDetails(
                    reasoning_tokens=50,
                ),
            ),
            {
                "llm.token_count.completion": 500,
                "llm.token_count.completion_details.reasoning": 50,
                "llm.token_count.prompt": 1000,
                "llm.token_count.prompt_details.cache_read": 100,
                "llm.token_count.total": 1500,
            },
            id="large_token_counts",
        ),
    ],
)
def test_get_attributes_from_usage(
    usage: ResponseUsage | None,
    expected_attributes: Mapping[str, Any],
) -> None:
    attributes = dict(_get_attributes_from_usage(usage))
    assert attributes == expected_attributes
