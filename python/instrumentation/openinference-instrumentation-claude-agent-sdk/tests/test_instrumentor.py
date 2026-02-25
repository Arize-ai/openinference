"""Tests for Claude Agent SDK instrumentor."""

from typing import Any

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation import OITracer
from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
)


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    provider = TracerProvider(resource=Resource(attributes={}))
    provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return provider


@pytest.fixture
def setup_instrumentation(
    tracer_provider: TracerProvider,
) -> Any:
    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    yield instrumentor
    instrumentor.uninstrument()


def test_entrypoint_opentelemetry_instrumentor() -> None:
    eps = entry_points(group="opentelemetry_instrumentor", name="claude_agent_sdk")  # type: ignore[no-untyped-call]
    (ep,) = eps
    instrumentor = ep.load()()
    assert isinstance(instrumentor, ClaudeAgentSDKInstrumentor)


def test_entrypoint_openinference_instrumentor() -> None:
    eps = entry_points(group="openinference_instrumentor", name="claude_agent_sdk")  # type: ignore[no-untyped-call]
    (ep,) = eps
    instrumentor = ep.load()()
    assert isinstance(instrumentor, ClaudeAgentSDKInstrumentor)


def test_oitracer(setup_instrumentation: Any) -> None:
    assert isinstance(setup_instrumentation._tracer, OITracer)


@pytest.mark.asyncio
async def test_query_span_created(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
) -> None:
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

    # Output messages: at least one message with role and content
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
    tracer_provider: TracerProvider,
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
    tracer_provider: TracerProvider,
) -> None:
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
    tracer_provider: TracerProvider,
) -> None:
    """When query() raises, span has status ERROR and exception is recorded."""
    import importlib

    from opentelemetry import trace as trace_api

    query_module = importlib.import_module("claude_agent_sdk.query")
    real_query = getattr(query_module, "query", None)
    if real_query is None:
        pytest.skip("claude_agent_sdk.query module has no 'query' attribute")

    async def failing_query(*, prompt: str = "", options: Any = None) -> Any:
        # Raise on first iteration so wrapper's async for sees the exception
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
    tracer_provider: TracerProvider,
) -> None:
    import importlib

    from claude_agent_sdk import ClaudeSDKClient
    from opentelemetry import trace as trace_api

    # Minimal ResultMessage-like object so message parsing works
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
        # Simulate that query("Hello") was just called
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
    # Real MCP tools (from @tool + create_sdk_mcp_server) and built-in tool captured
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
    tracer_provider: TracerProvider,
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
