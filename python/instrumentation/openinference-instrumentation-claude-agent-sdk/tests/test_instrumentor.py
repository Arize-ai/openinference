"""Tests for Claude Agent SDK instrumentor.

Unit tests use mocks (no HTTP). The integration test (test_query_real_agent_span)
runs against the real Anthropic API and is skipped unless a real ANTHROPIC_API_KEY
is provided in the environment.
"""

from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from openinference.instrumentation import OITracer
from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
)

TOOL_KIND = OpenInferenceSpanKindValues.TOOL.value


def test_oitracer(tracer_provider: Any) -> None:
    """Instrument uses OITracer."""
    inst = ClaudeAgentSDKInstrumentor()
    inst.instrument(tracer_provider=tracer_provider)
    try:
        assert isinstance(inst._tracer, OITracer)
    finally:
        inst.uninstrument()


# ---- Unit tests (mocked query / client, no HTTP) ----


@pytest.mark.asyncio
async def test_query_span_created(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """Mocked query() produces one AGENT span with input/output."""
    import importlib

    from opentelemetry import trace as trace_api

    class SystemMessage:
        subtype = "init"
        data: dict[str, Any] = {}

    query_module = importlib.import_module("claude_agent_sdk.query")
    real_query = getattr(query_module, "query", None)
    if real_query is None:
        pytest.skip("claude_agent_sdk.query module has no 'query' attribute")

    async def fake_query(*, prompt: str = "", options: Any = None) -> Any:
        yield SystemMessage()

    setattr(query_module, "query", fake_query)
    trace_api.set_tracer_provider(tracer_provider)
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)

    try:
        count = 0
        query_fn = getattr(query_module, "query")
        async for _ in query_fn(prompt="Hello", options=None):
            count += 1
        assert count == 1
    finally:
        ClaudeAgentSDKInstrumentor().uninstrument()
        setattr(query_module, "query", real_query)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "ClaudeAgentSDK.query"
    attrs = dict(span.attributes or {})
    assert (
        attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.AGENT.value
    )
    assert SpanAttributes.INPUT_VALUE in attrs
    assert SpanAttributes.INPUT_MIME_TYPE in attrs
    assert SpanAttributes.OUTPUT_VALUE in attrs
    assert SpanAttributes.OUTPUT_MIME_TYPE in attrs
    assert "Hello" in str(attrs.get(SpanAttributes.INPUT_VALUE, ""))

    role_key = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"
    assert role_key in attrs
    assert attrs[role_key] == "system"
    content_key = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"
    assert content_key in attrs
    assert "init" in str(attrs[content_key])


def _make_mcp_calculator_options() -> Any:
    """Build ClaudeAgentOptions with a real SDK MCP server (@tool + create_sdk_mcp_server)."""
    from claude_agent_sdk import create_sdk_mcp_server, tool
    from claude_agent_sdk.types import ClaudeAgentOptions

    @tool("add", "Add two numbers", {"a": float, "b": float})
    async def add(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": str(args["a"] + args["b"])}]}

    @tool("multiply", "Multiply two numbers", {"a": float, "b": float})
    async def multiply(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": str(args["a"] * args["b"])}]}

    calc_server = create_sdk_mcp_server(
        name="calc",
        version="1.0.0",
        tools=[add, multiply],
    )
    return ClaudeAgentOptions(
        mcp_servers={"calc": calc_server},
        allowed_tools=["mcp__calc__add", "mcp__calc__multiply", "Bash"],
    )


@pytest.mark.asyncio
async def test_tools_captured(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """When options has allowed_tools, llm.tools attributes are set on the span."""
    import importlib

    from claude_agent_sdk.types import ClaudeAgentOptions
    from opentelemetry import trace as trace_api

    class SystemMessage:
        subtype = "init"
        data: dict[str, Any] = {}

    query_module = importlib.import_module("claude_agent_sdk.query")
    real_query = getattr(query_module, "query", None)
    if real_query is None:
        pytest.skip("claude_agent_sdk.query module has no 'query' attribute")

    async def fake_query(*, prompt: str = "", options: Any = None) -> Any:
        yield SystemMessage()

    setattr(query_module, "query", fake_query)
    trace_api.set_tracer_provider(tracer_provider)
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)

    try:
        options = ClaudeAgentOptions(allowed_tools=["Bash", "Glob"])
        query_fn = getattr(query_module, "query")
        async for _ in query_fn(prompt="List files", options=options):
            pass
    finally:
        ClaudeAgentSDKInstrumentor().uninstrument()
        setattr(query_module, "query", real_query)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    tool0 = attrs.get(f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}")
    tool1 = attrs.get(f"{SpanAttributes.LLM_TOOLS}.1.{ToolAttributes.TOOL_JSON_SCHEMA}")
    assert tool0 is not None
    assert tool1 is not None
    assert "Bash" in str(tool0)
    assert "Glob" in str(tool1)


@pytest.mark.asyncio
async def test_mcp_tools_captured(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """MCP tools from options are recorded on the span."""
    import importlib

    from opentelemetry import trace as trace_api

    class SystemMessage:
        subtype = "init"
        data: dict[str, Any] = {}

    query_module = importlib.import_module("claude_agent_sdk.query")
    real_query = getattr(query_module, "query", None)
    if real_query is None:
        pytest.skip("claude_agent_sdk.query module has no 'query' attribute")

    async def fake_query(*, prompt: str = "", options: Any = None) -> Any:
        yield SystemMessage()

    setattr(query_module, "query", fake_query)
    trace_api.set_tracer_provider(tracer_provider)
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)

    try:
        options = _make_mcp_calculator_options()
        query_fn = getattr(query_module, "query")
        async for _ in query_fn(prompt="Use the calculator", options=options):
            pass
    finally:
        ClaudeAgentSDKInstrumentor().uninstrument()
        setattr(query_module, "query", real_query)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    tool_schemas = [
        attrs.get(f"{SpanAttributes.LLM_TOOLS}.{i}.{ToolAttributes.TOOL_JSON_SCHEMA}")
        for i in range(3)
    ]
    assert all(tool_schemas)
    names = " ".join(str(s) for s in tool_schemas)
    assert "mcp__calc__add" in names
    assert "mcp__calc__multiply" in names
    assert "Bash" in names


@pytest.mark.asyncio
async def test_error_recorded(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """When query() raises, span has status ERROR and exception is recorded."""
    import importlib

    from opentelemetry import trace as trace_api

    query_module = importlib.import_module("claude_agent_sdk.query")
    real_query = getattr(query_module, "query", None)
    if real_query is None:
        pytest.skip("claude_agent_sdk.query module has no 'query' attribute")

    async def failing_query(*, prompt: str = "", options: Any = None) -> Any:
        raise RuntimeError("Simulated failure")
        yield  # make this an async generator; unreachable

    setattr(query_module, "query", failing_query)
    trace_api.set_tracer_provider(tracer_provider)
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)

    try:
        query_fn = getattr(query_module, "query")
        with pytest.raises(RuntimeError, match="Simulated failure"):
            async for _ in query_fn(prompt="Hello", options=None):
                pass
    finally:
        ClaudeAgentSDKInstrumentor().uninstrument()
        setattr(query_module, "query", real_query)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert "RuntimeError" in str(span.status.description)
    assert "Simulated failure" in str(span.status.description)


@pytest.mark.asyncio
async def test_client_receive_response_span_created(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """ClaudeSDKClient.receive_response produces one AGENT span per turn."""
    import importlib

    from claude_agent_sdk import ClaudeSDKClient
    from opentelemetry import trace as trace_api

    class MinimalResultMessage:
        subtype = "success"
        result = "Done"
        data: dict[str, Any] = {}

    client_module = importlib.import_module("claude_agent_sdk.client")
    real_receive = client_module.ClaudeSDKClient.receive_response

    async def fake_receive(self: Any) -> Any:
        yield MinimalResultMessage()

    client_module.ClaudeSDKClient.receive_response = fake_receive
    trace_api.set_tracer_provider(tracer_provider)
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)

    try:
        options = _make_mcp_calculator_options()
        client = ClaudeSDKClient(options=options)
        setattr(client, "_oinference_last_prompt", "Hello")
        setattr(client, "_oinference_last_options", options)
        async for _ in client.receive_response():
            pass
    finally:
        ClaudeAgentSDKInstrumentor().uninstrument()
        client_module.ClaudeSDKClient.receive_response = real_receive

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "ClaudeAgentSDK.ClaudeSDKClient.receive_response"
    attrs = dict(span.attributes or {})
    assert (
        attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.AGENT.value
    )
    assert "Hello" in str(attrs.get(SpanAttributes.INPUT_VALUE, ""))
    tool_str = " ".join(
        str(attrs.get(f"{SpanAttributes.LLM_TOOLS}.{i}.{ToolAttributes.TOOL_JSON_SCHEMA}"))
        for i in range(3)
    )
    assert "mcp__calc__add" in tool_str
    assert "mcp__calc__multiply" in tool_str
    assert "Bash" in tool_str


@pytest.mark.asyncio
async def test_usage_captured_from_result_message(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """Usage from ResultMessage.usage is recorded as llm.token_count.* on the span."""
    import importlib

    from claude_agent_sdk import ClaudeSDKClient
    from opentelemetry import trace as trace_api

    class ResultMessageWithUsage:
        subtype = "success"
        result = "42"
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 10,
            "cache_creation_input_tokens": 5,
        }
        data: dict[str, Any] = {}

    client_module = importlib.import_module("claude_agent_sdk.client")
    real_receive = client_module.ClaudeSDKClient.receive_response

    async def fake_receive(self: Any) -> Any:
        yield ResultMessageWithUsage()

    client_module.ClaudeSDKClient.receive_response = fake_receive
    trace_api.set_tracer_provider(tracer_provider)
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)

    try:
        client = ClaudeSDKClient(options=None)
        setattr(client, "_oinference_last_prompt", "What is 2+2?")
        setattr(client, "_oinference_last_options", None)
        async for _ in client.receive_response():
            pass
    finally:
        ClaudeAgentSDKInstrumentor().uninstrument()
        client_module.ClaudeSDKClient.receive_response = real_receive

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 100
    assert attrs.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 50
    assert attrs.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 150
    assert attrs.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == 10
    assert attrs.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE) == 5


# ---- Tool span tests (message-based) ----


class _ToolUseBlock:
    """Minimal ToolUseBlock mock matching claude_agent_sdk.types.ToolUseBlock."""

    def __init__(self, id: str, name: str, input: dict[str, Any]) -> None:
        self.id = id
        self.name = name
        self.input = input


class _ToolResultBlock:
    """Minimal ToolResultBlock mock matching claude_agent_sdk.types.ToolResultBlock."""

    def __init__(self, tool_use_id: str, content: Any = None, is_error: bool | None = None) -> None:
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error


class _AssistantMessage:
    """Minimal AssistantMessage mock."""

    def __init__(self, content: list[Any]) -> None:
        self.content = content
        self.model = "claude-3"
        self.parent_tool_use_id = None
        self.error = None


class _UserMessage:
    """Minimal UserMessage mock."""

    def __init__(self, content: list[Any]) -> None:
        self.content = content
        self.uuid = None
        self.parent_tool_use_id = None
        self.tool_use_result = None


class _ResultMessage:
    """Minimal ResultMessage mock."""

    def __init__(self, result: str) -> None:
        self.result = result
        self.subtype = "success"
        self.usage = None


@pytest.mark.asyncio
async def test_tool_spans_created_from_messages(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """AssistantMessage/UserMessage with tool blocks produce TOOL child spans."""
    import importlib

    from opentelemetry import trace as trace_api

    query_module = importlib.import_module("claude_agent_sdk.query")
    real_query = getattr(query_module, "query", None)
    if real_query is None:
        pytest.skip("claude_agent_sdk.query module has no 'query' attribute")

    async def fake_query(*, prompt: str = "", options: Any = None) -> Any:
        # Assistant decides to call Bash
        yield _AssistantMessage(
            content=[_ToolUseBlock(id="tool-1", name="Bash", input={"command": "ls -la"})]
        )
        # Tool result comes back via UserMessage
        yield _UserMessage(
            content=[
                _ToolResultBlock(
                    tool_use_id="tool-1", content="file1.txt\nfile2.txt", is_error=False
                )
            ]
        )
        yield _ResultMessage(result="There are 2 files.")

    setattr(query_module, "query", fake_query)
    trace_api.set_tracer_provider(tracer_provider)
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)

    try:
        query_fn = getattr(query_module, "query")
        async for _ in query_fn(prompt="List files", options=None):
            pass
    finally:
        ClaudeAgentSDKInstrumentor().uninstrument()
        setattr(query_module, "query", real_query)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2, (
        f"Expected 2 spans (AGENT + TOOL), got {len(spans)}: {[s.name for s in spans]}"
    )

    span_by_name = {s.name: s for s in spans}
    assert "ClaudeAgentSDK.query" in span_by_name
    assert "Bash" in span_by_name

    tool_span = span_by_name["Bash"]
    attrs = dict(tool_span.attributes or {})
    assert attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == TOOL_KIND
    assert "ls -la" in str(attrs.get(SpanAttributes.INPUT_VALUE, ""))
    assert "file1.txt" in str(attrs.get(SpanAttributes.OUTPUT_VALUE, ""))
    assert tool_span.status.status_code == StatusCode.OK

    # Tool span must be a child of the AGENT span
    agent_span = span_by_name["ClaudeAgentSDK.query"]
    assert tool_span.context.trace_id == agent_span.context.trace_id
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == agent_span.context.span_id


@pytest.mark.asyncio
async def test_tool_span_error_from_message(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """ToolResultBlock with is_error=True produces a TOOL span with ERROR status."""
    import importlib

    from opentelemetry import trace as trace_api

    query_module = importlib.import_module("claude_agent_sdk.query")
    real_query = getattr(query_module, "query", None)
    if real_query is None:
        pytest.skip("claude_agent_sdk.query module has no 'query' attribute")

    async def fake_query(*, prompt: str = "", options: Any = None) -> Any:
        yield _AssistantMessage(
            content=[_ToolUseBlock(id="tool-err", name="Bash", input={"command": "rm -rf /"})]
        )
        yield _UserMessage(
            content=[
                _ToolResultBlock(tool_use_id="tool-err", content="Permission denied", is_error=True)
            ]
        )
        yield _ResultMessage(result="I could not complete that.")

    setattr(query_module, "query", fake_query)
    trace_api.set_tracer_provider(tracer_provider)
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)

    try:
        query_fn = getattr(query_module, "query")
        async for _ in query_fn(prompt="Run dangerous command", options=None):
            pass
    finally:
        ClaudeAgentSDKInstrumentor().uninstrument()
        setattr(query_module, "query", real_query)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2, f"Expected 2 spans, got {[s.name for s in spans]}"

    span_by_name = {s.name: s for s in spans}
    tool_span = span_by_name["Bash"]
    assert tool_span.status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_multiple_tool_calls_produce_multiple_spans(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """Multiple tool calls in a sequence each get their own TOOL span."""
    import importlib

    from opentelemetry import trace as trace_api

    query_module = importlib.import_module("claude_agent_sdk.query")
    real_query = getattr(query_module, "query", None)
    if real_query is None:
        pytest.skip("claude_agent_sdk.query module has no 'query' attribute")

    async def fake_query(*, prompt: str = "", options: Any = None) -> Any:
        # First tool call
        yield _AssistantMessage(
            content=[_ToolUseBlock(id="t1", name="Bash", input={"command": "ls"})]
        )
        yield _UserMessage(
            content=[_ToolResultBlock(tool_use_id="t1", content="file.txt", is_error=False)]
        )
        # Second tool call
        yield _AssistantMessage(
            content=[_ToolUseBlock(id="t2", name="Glob", input={"pattern": "*.py"})]
        )
        yield _UserMessage(
            content=[_ToolResultBlock(tool_use_id="t2", content="main.py", is_error=False)]
        )
        yield _ResultMessage(result="Done.")

    setattr(query_module, "query", fake_query)
    trace_api.set_tracer_provider(tracer_provider)
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)

    try:
        query_fn = getattr(query_module, "query")
        async for _ in query_fn(prompt="Find files", options=None):
            pass
    finally:
        ClaudeAgentSDKInstrumentor().uninstrument()
        setattr(query_module, "query", real_query)

    spans = in_memory_span_exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    assert len(spans) == 3, f"Expected AGENT + 2 TOOL spans, got: {span_names}"
    assert "ClaudeAgentSDK.query" in span_names
    assert "Bash" in span_names
    assert "Glob" in span_names

    agent_span = next(s for s in spans if s.name == "ClaudeAgentSDK.query")
    for s in spans:
        if s.name in ("Bash", "Glob"):
            assert s.parent is not None
            assert s.parent.span_id == agent_span.context.span_id


@pytest.mark.asyncio
async def test_tool_calls_captured_as_output_message_attributes(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """Tool use blocks in AssistantMessage are also recorded as llm.output_messages tool_calls."""
    import importlib

    from opentelemetry import trace as trace_api

    query_module = importlib.import_module("claude_agent_sdk.query")
    real_query = getattr(query_module, "query", None)
    if real_query is None:
        pytest.skip("claude_agent_sdk.query module has no 'query' attribute")

    async def fake_query(*, prompt: str = "", options: Any = None) -> Any:
        yield _AssistantMessage(
            content=[_ToolUseBlock(id="t1", name="Bash", input={"command": "echo hello"})]
        )
        yield _UserMessage(
            content=[_ToolResultBlock(tool_use_id="t1", content="hello", is_error=False)]
        )
        yield _ResultMessage(result="Done.")

    setattr(query_module, "query", fake_query)
    trace_api.set_tracer_provider(tracer_provider)
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)

    try:
        query_fn = getattr(query_module, "query")
        async for _ in query_fn(prompt="Say hello", options=None):
            pass
    finally:
        ClaudeAgentSDKInstrumentor().uninstrument()
        setattr(query_module, "query", real_query)

    spans = in_memory_span_exporter.get_finished_spans()
    agent_span = next(s for s in spans if s.name == "ClaudeAgentSDK.query")
    attrs = dict(agent_span.attributes or {})

    # Tool call function name should appear in llm.output_messages attributes
    found_tool_name = any("Bash" in str(v) for v in attrs.values())
    assert found_tool_name, f"Expected 'Bash' in span attributes, got: {list(attrs.keys())}"


# ---- Integration test: real HTTP against the live Anthropic API ----
#
# Skipped automatically when no real ANTHROPIC_API_KEY is present (CI uses the
# fake key set by the api_key autouse fixture in conftest.py).


@pytest.mark.asyncio
async def test_query_real_agent_span(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Real query() call produces one AGENT span.

    Requires a real ANTHROPIC_API_KEY in the environment. Skipped otherwise.
    """
    import os

    if os.environ.get("ANTHROPIC_API_KEY") == "test-key-no-real-call":
        pytest.skip("Requires a real ANTHROPIC_API_KEY to run integration test")

    from claude_agent_sdk import query

    prompt = "Reply with exactly the word: ok"
    async for _ in query(prompt=prompt):
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "ClaudeAgentSDK.query"
    assert span.status.is_ok
    attrs = dict(span.attributes or {})
    assert (
        attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.AGENT.value
    )
    assert SpanAttributes.INPUT_VALUE in attrs
    assert prompt in str(attrs.get(SpanAttributes.INPUT_VALUE, ""))
    assert SpanAttributes.OUTPUT_VALUE in attrs
    assert str(attrs.get(SpanAttributes.OUTPUT_VALUE, ""))


@pytest.mark.asyncio
async def test_client_real_agent_span(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """ClaudeSDKClient.query() + receive_response() produces one AGENT span.

    Requires a real ANTHROPIC_API_KEY in the environment. Skipped otherwise.
    """
    import os

    if os.environ.get("ANTHROPIC_API_KEY") == "test-key-no-real-call":
        pytest.skip("Requires a real ANTHROPIC_API_KEY to run integration test")

    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    prompt = "Reply with exactly the word: ok"
    options = ClaudeAgentOptions(allowed_tools=["Bash"])
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for _ in client.receive_response():
            pass

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "ClaudeAgentSDK.ClaudeSDKClient.receive_response"
    assert span.status.is_ok
    attrs = dict(span.attributes or {})
    assert (
        attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.AGENT.value
    )
    assert SpanAttributes.INPUT_VALUE in attrs
    assert prompt in str(attrs.get(SpanAttributes.INPUT_VALUE, ""))
    assert SpanAttributes.OUTPUT_VALUE in attrs
    assert str(attrs.get(SpanAttributes.OUTPUT_VALUE, ""))
