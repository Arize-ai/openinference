"""Calls that OpenAI makes for the model on a remote MCP server or in a code interpreter.

A Responses API reply returns them as `mcp_call`, `mcp_approval_request` and
`code_interpreter_call` output items, which were left out of the output messages, so the span had
no tool call for them.
"""

import asyncio
import importlib
import json
from typing import Any, Dict, Mapping, cast

import pytest
from httpx import Response
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue
from respx import MockRouter

from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)

_BASE_URL = "https://api.openai.com/v1/"
_MODEL = "gpt-4o-mini"
_INPUT = "Search the docs, then compute 1 + 1"
_MCP_TOOL = {
    "type": "mcp",
    "server_label": "docs",
    "server_url": "https://mcp.example.com/sse",
    "require_approval": "never",
}
_OUTPUT = [
    {
        "id": "mcpl_1",
        "type": "mcp_list_tools",
        "server_label": "docs",
        "tools": [{"name": "search_docs", "input_schema": {"type": "object"}}],
    },
    {
        "id": "mcp_1",
        "type": "mcp_call",
        "server_label": "docs",
        "name": "search_docs",
        "arguments": '{"query": "spans"}',
        "output": "3 results",
        "error": None,
        "status": "completed",
    },
    {
        "id": "mcpr_1",
        "type": "mcp_approval_request",
        "server_label": "docs",
        "name": "delete_docs",
        "arguments": "{}",
    },
    {
        "id": "ci_1",
        "type": "code_interpreter_call",
        "container_id": "cntr_1",
        "code": "print(1 + 1)",
        "outputs": [{"type": "logs", "logs": "2"}],
        "status": "completed",
    },
    {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Found three; 1 + 1 = 2.", "annotations": []}],
    },
]


@pytest.mark.parametrize("is_async", [False, True])
def test_hosted_tool_calls_are_recorded_as_tool_calls(
    is_async: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    respx_mock.post(f"{_BASE_URL}responses").mock(
        return_value=Response(
            status_code=200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 1758000000,
                "model": _MODEL,
                "output": _OUTPUT,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [_MCP_TOOL],
            },
        )
    )
    openai = importlib.import_module("openai")
    request: Dict[str, Any] = {"model": _MODEL, "input": _INPUT, "tools": [_MCP_TOOL]}
    if is_async:
        client = openai.AsyncOpenAI(api_key="sk-", base_url=_BASE_URL)
        asyncio.run(client.responses.create(**request))
    else:
        openai.OpenAI(api_key="sk-", base_url=_BASE_URL).responses.create(**request)

    (span,) = [
        span
        for span in in_memory_span_exporter.get_finished_spans()
        if span.instrumentation_scope is not None
        and span.instrumentation_scope.name == "openinference.instrumentation.openai"
    ]
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.LLM.value
    )
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "openai"
    assert attributes.pop(SpanAttributes.LLM_SYSTEM) == "openai"
    assert json.loads(cast(str, attributes.pop(SpanAttributes.INPUT_VALUE))) == request
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == OpenInferenceMimeTypeValues.JSON.value
    assert json.loads(cast(str, attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS))) == {
        "model": _MODEL
    }
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == _MODEL
    assert attributes.pop(f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.message.role") == "user"
    assert attributes.pop(f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.message.content") == _INPUT
    # the response's copy of the tool, with every field of the SDK's model
    tool = json.loads(cast(str, attributes.pop("llm.tools.0.tool.json_schema")))
    assert {key: tool[key] for key in _MCP_TOOL} == _MCP_TOOL
    assert isinstance(attributes.pop(SpanAttributes.OUTPUT_VALUE), str)
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == OpenInferenceMimeTypeValues.JSON.value

    # The MCP server's tool list, at index 0, is not a tool call. An approval request with no
    # arguments records none, as a function call with none does.
    for i, tool_call_id, name, arguments in [
        (1, "mcp_1", "search_docs", '{"query": "spans"}'),
        (2, "mcpr_1", "delete_docs", None),
        (3, "ci_1", "code_interpreter_call", '{"code": "print(1 + 1)"}'),
    ]:
        message = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{i}."
        tool_call = f"{message}{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
        assert attributes.pop(f"{message}{MessageAttributes.MESSAGE_ROLE}") == "assistant"
        assert attributes.pop(f"{tool_call}{ToolCallAttributes.TOOL_CALL_ID}") == tool_call_id
        assert attributes.pop(f"{tool_call}{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}") == name
        assert (
            attributes.pop(
                f"{tool_call}{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}", None
            )
            == arguments
        )
    message = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.4."
    content = f"{message}{MessageAttributes.MESSAGE_CONTENTS}.0."
    assert attributes.pop(f"{message}{MessageAttributes.MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(f"{content}{MessageContentAttributes.MESSAGE_CONTENT_TYPE}") == "text"
    assert attributes.pop(f"{content}{MessageContentAttributes.MESSAGE_CONTENT_TEXT}") == (
        "Found three; 1 + 1 = 2."
    )
    assert attributes == {}
