"""
Tests against recorded responses of the Messages API features that are traced, e.g. server tools,
citations, token counting, inline tool changes and programmatic tool calling. To re-record, delete
the cassette and run the test with ANTHROPIC_API_KEY set.
"""

import json
import os
from typing import Any, Callable, Dict, Iterator, List

import pytest
from anthropic import Anthropic, AsyncAnthropic
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

MODEL = "claude-haiku-5-5"
WEB_SEARCH_TOOL: Any = {"type": "web_search_20250305", "name": "web_search", "max_uses": 1}
WEB_SEARCH_PROMPT = "Using web search, what year did the Sydney Opera House open? Cite a source."

LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_TOOLS = SpanAttributes.LLM_TOOLS
METADATA = SpanAttributes.METADATA
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA


def _api_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY", "sk-ant-fake")


def _get_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    (span,) = exporter.get_finished_spans()
    return span


def _get_attributes(exporter: InMemorySpanExporter) -> Dict[str, Any]:
    span = _get_span(exporter)
    assert span.status.is_ok
    return dict(span.attributes or {})


def _indexed(attributes: Dict[str, Any], prefix: str, suffix: str) -> Iterator[Any]:
    index = 0
    while (key := f"{prefix}.{index}.{suffix}") in attributes:
        yield attributes[key]
        index += 1


def _tool_names(attributes: Dict[str, Any]) -> List[str]:
    return [
        json.loads(str(tool))["name"] for tool in _indexed(attributes, LLM_TOOLS, TOOL_JSON_SCHEMA)
    ]


def _assert_web_search_span(attributes: Dict[str, Any]) -> None:
    message = f"{LLM_OUTPUT_MESSAGES}.0"
    (tool_call_id,) = _indexed(attributes, f"{message}.{MESSAGE_TOOL_CALLS}", TOOL_CALL_ID)
    assert tool_call_id.startswith("srvtoolu_")
    tool_call = f"{message}.{MESSAGE_TOOL_CALLS}.0"
    assert attributes[f"{tool_call}.{TOOL_CALL_FUNCTION_NAME}"] == "web_search"
    assert json.loads(attributes[f"{tool_call}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"])["query"]
    content_types = list(
        _indexed(attributes, f"{message}.{MESSAGE_CONTENTS}", MESSAGE_CONTENT_TYPE)
    )
    # the server tool call and its result are recorded in the order of their blocks
    result_index = content_types.index("web_search_tool_result")
    assert content_types[result_index - 1] == "tool_use"
    result = json.loads(
        attributes[f"{message}.{MESSAGE_CONTENTS}.{result_index}.{MESSAGE_CONTENT_TEXT}"]
    )
    assert result["tool_use_id"] == tool_call_id
    metadata = json.loads(attributes[METADATA])
    assert metadata["server_tool_use"]["web_search_requests"] == 1
    citations = [citation for block in metadata["citations"].values() for citation in block]
    assert citations
    assert all(citation["url"].startswith("https://") for citation in citations)
    assert all(citation["cited_text"] for citation in citations)


@pytest.mark.vcr
def test_web_search_server_tool_calls(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = Anthropic(api_key=_api_key())
    client.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": WEB_SEARCH_PROMPT}],
    )
    _assert_web_search_span(_get_attributes(in_memory_span_exporter))


@pytest.mark.vcr
def test_streaming_web_search_server_tool_calls(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = Anthropic(api_key=_api_key())
    with client.messages.stream(
        model=MODEL,
        max_tokens=1024,
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": WEB_SEARCH_PROMPT}],
    ) as stream:
        stream.until_done()
    _assert_web_search_span(_get_attributes(in_memory_span_exporter))


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


@pytest.mark.default_cassette("test_web_search_server_tool_calls.yaml")
@pytest.mark.vcr(record_mode="none")
@pytest.mark.parametrize(
    "config, hides_citations, hides_cited_text",
    [
        ({}, False, False),
        ({"hide_inputs": True}, True, True),
        ({"hide_input_messages": True}, True, True),
        ({"hide_input_text": True}, False, True),
        ({"hide_outputs": True}, True, True),
        ({"hide_output_messages": True}, True, True),
        ({"hide_output_text": True}, False, True),
    ],
)
def test_web_search_citations_respect_trace_config(
    in_memory_span_exporter: InMemorySpanExporter,
    instrument_with_config: Callable[..., None],
    config: Dict[str, Any],
    hides_citations: bool,
    hides_cited_text: bool,
) -> None:
    instrument_with_config(**config)
    client = Anthropic(api_key=_api_key())
    client.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": WEB_SEARCH_PROMPT}],
    )
    metadata = json.loads(_get_attributes(in_memory_span_exporter)[METADATA])
    assert metadata["server_tool_use"]["web_search_requests"] == 1
    if hides_citations:
        assert "citations" not in metadata
        return
    citations = [citation for block in metadata["citations"].values() for citation in block]
    assert citations
    assert all(("cited_text" in citation) is not hides_cited_text for citation in citations)
    assert all(citation["url"].startswith("https://") for citation in citations)


_COUNT_TOKENS_KWARGS: Dict[str, Any] = {
    "model": MODEL,
    "system": "Be brief.",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
}


def _assert_count_tokens_span(attributes: Dict[str, Any], input_tokens: int) -> None:
    assert json.loads(attributes.pop(SpanAttributes.OUTPUT_VALUE))["input_tokens"] == input_tokens
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == MODEL
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENT}") == (
        "What is the capital of France?"
    )
    assert _tool_names(attributes) == ["get_weather"]
    # nothing is generated, so there are no token counts
    assert not [key for key in attributes if key.startswith("llm.token_count")]


@pytest.mark.vcr
@pytest.mark.parametrize("is_beta", [False, True])
def test_count_tokens(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
    is_beta: bool,
) -> None:
    client = Anthropic(api_key=_api_key())
    messages: Any = client.beta.messages if is_beta else client.messages
    input_tokens = messages.count_tokens(**_COUNT_TOKENS_KWARGS).input_tokens
    assert input_tokens > 0
    _assert_count_tokens_span(_get_attributes(in_memory_span_exporter), input_tokens)


@pytest.mark.vcr
async def test_async_count_tokens(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = AsyncAnthropic(api_key=_api_key())
    input_tokens = (await client.messages.count_tokens(**_COUNT_TOKENS_KWARGS)).input_tokens
    assert input_tokens > 0
    _assert_count_tokens_span(_get_attributes(in_memory_span_exporter), input_tokens)


def _tool(name: str) -> Any:
    return {
        "name": name,
        "description": f"The {name.replace('_', ' ')}.",
        "input_schema": {"type": "object", "properties": {}},
    }


@pytest.mark.vcr
def test_tool_changes_in_system_messages_are_applied_to_tools(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    # tool changes are sent in a system message that follows a user message
    messages: Any = [
        {"role": "user", "content": "Which tools can you use? Answer in one line."},
        {
            "role": "system",
            "content": [
                {"type": "tool_removal", "tool": {"type": "tool_reference", "name": "get_date"}},
                {
                    "type": "tool_addition",
                    "tool": {"type": "tool_definition", "definition": _tool("get_weather")},
                },
            ],
        },
    ]
    client = Anthropic(api_key=_api_key())
    client.beta.messages.create(
        model=MODEL,
        max_tokens=256,
        betas=["inline-tools-2026-09-15"],
        tools=[_tool("get_time"), _tool("get_date")],
        messages=messages,
    )
    attributes = _get_attributes(in_memory_span_exporter)
    assert _tool_names(attributes) == ["get_time", "get_weather"]
    contents = f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}"
    assert list(_indexed(attributes, contents, MESSAGE_CONTENT_TYPE)) == [
        "tool_removal",
        "tool_addition",
    ]


@pytest.mark.vcr
def test_programmatic_tool_calls_record_their_caller(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = Anthropic(api_key=_api_key())
    get_weather: Any = {
        **_tool("get_weather"),
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        "allowed_callers": ["code_execution_20260120"],
    }
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=[{"type": "code_execution_20260120", "name": "code_execution"}, get_weather],
        messages=[
            {
                "role": "user",
                "content": "Write code that calls get_weather for Paris, then report the result.",
            }
        ],
    )
    (tool_use,) = [block for block in response.content if block.type == "tool_use"]
    metadata = json.loads(_get_attributes(in_memory_span_exporter)[METADATA])
    caller = metadata["tool_callers"][tool_use.id]
    assert caller["type"] == "code_execution_20260120"
    assert caller["tool_id"].startswith("srvtoolu_")
