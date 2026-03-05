"""Tests for Claude Agent SDK instrumentor.

Unit tests use mocks (no HTTP). Integration tests use pre-recorded cassettes via
cassette_transport (always run in CI, no real API key needed).
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from openinference.instrumentation import OITracer
from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

TOOL_KIND = OpenInferenceSpanKindValues.TOOL.value
AGENT_KIND = OpenInferenceSpanKindValues.AGENT.value


def _span_by_name(spans: Sequence[Any], name: str) -> Any:
    for span in spans:
        if span.name == name:
            return span
    raise AssertionError(f"Span not found: {name}. Names: {[s.name for s in spans]}")


def _tool_spans(spans: Sequence[Any]) -> list[Any]:
    tool_spans: list[Any] = []
    for span in spans:
        attrs = dict(span.attributes or {})
        if attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == TOOL_KIND:
            tool_spans.append(span)
    return tool_spans


class _DummyOptions:
    def __init__(self) -> None:
        self.hooks: dict[str, Any] = {}


async def _run_hooks(options: Any, event: str, payload: dict[str, Any]) -> None:
    hooks = getattr(options, "hooks", {}) if options is not None else {}
    matchers = hooks.get(event, []) if isinstance(hooks, dict) else []
    if not isinstance(matchers, list):
        matchers = [matchers]
    for matcher in matchers:
        callbacks = None
        if isinstance(matcher, dict):
            callbacks = matcher.get("hooks")
        else:
            callbacks = getattr(matcher, "hooks", None)
        if not callbacks:
            continue
        for cb in callbacks:
            result = cb(payload)
            if inspect.isawaitable(result):
                await result


def _hook_matcher(callback: Callable[[Any], Any]) -> Any:
    for module_path, name in (
        ("claude_agent_sdk", "HookMatcher"),
        ("claude_agent_sdk.types", "HookMatcher"),
    ):
        try:
            module = __import__(module_path, fromlist=[name])
            matcher_type = getattr(module, name)
        except Exception:
            matcher_type = None
        if matcher_type is None:
            continue
        try:
            return matcher_type(hooks=[callback])
        except Exception:
            continue
    return {"hooks": [callback]}


def _payload_field(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        return payload.get(key)
    return getattr(payload, key, None)


_PYPROJECT_PATH = (Path(__file__).resolve().parent.parent / "pyproject.toml").as_posix()
_TOOL_PROMPT = (
    "Use the Bash tool to run: wc -c '"
    f"{_PYPROJECT_PATH}"
    "' and respond with exactly the output. Do not answer unless you executed the tool."
)
_TASK_PROMPT = (
    "You must use the Agent tool to delegate to a subagent. "
    "The subagent must use the Bash tool to run: wc -c '"
    f"{_PYPROJECT_PATH}"
    "' and respond with exactly the output. Do not answer unless you executed the tool."
)


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
async def test_result_error_message_sets_status_and_output(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """Result error message marks span ERROR and captures error output."""
    import importlib

    from opentelemetry import trace as trace_api

    class ResultErrorMessage:
        type = "result"
        subtype = "error:tool"
        errors = [{"code": "tool_error", "message": "Permission denied"}]
        usage = {"input_tokens": 3, "output_tokens": 1}
        total_cost_usd = 0.02
        session_id = "sess-err"

    query_module = importlib.import_module("claude_agent_sdk.query")
    real_query = getattr(query_module, "query", None)
    if real_query is None:
        pytest.skip("claude_agent_sdk.query module has no 'query' attribute")

    async def fake_query(*, prompt: str = "", options: Any = None) -> Any:
        yield ResultErrorMessage()

    setattr(query_module, "query", fake_query)
    trace_api.set_tracer_provider(tracer_provider)
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)

    try:
        query_fn = getattr(query_module, "query")
        async for _ in query_fn(prompt="Hello", options=None):
            pass
    finally:
        ClaudeAgentSDKInstrumentor().uninstrument()
        setattr(query_module, "query", real_query)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})

    assert span.status.status_code == StatusCode.ERROR
    assert "tool_error" in str(attrs.get(SpanAttributes.OUTPUT_VALUE, ""))
    assert attrs.get(SpanAttributes.SESSION_ID) == "sess-err"
    assert attrs.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 3
    assert attrs.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 1
    assert attrs.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 4
    assert attrs.get(SpanAttributes.LLM_COST_TOTAL) == 0.02


@pytest.mark.asyncio
async def test_query_exception_sets_error_status(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """Exceptions from query() are recorded and set ERROR status."""
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


def test_merge_hooks_preserves_user_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    """User hooks should be preserved when instrumentation merges hooks."""
    import openinference.instrumentation.claude_agent_sdk._wrappers as wrappers

    user_called: list[str] = []

    def user_hook(payload: Any) -> dict[str, Any]:
        user_called.append(_payload_field(payload, "tool_use_id") or "")
        return {}

    def fake_create_tool_hook_matchers(_: Any) -> dict[str, list[Any]]:
        async def noop(_: Any) -> dict[str, Any]:
            return {}

        return {"PreToolUse": [{"hooks": [noop]}]}

    monkeypatch.setattr(wrappers, "_create_tool_hook_matchers", fake_create_tool_hook_matchers)

    class _NoopTracker(wrappers._ToolSpanTrackerBase):
        def start_tool_span(
            self, tool_name: Any, tool_input: Any, tool_use_id: Any, parent_tool_use_id: Any = None
        ) -> None:
            return None

        def end_tool_span(self, tool_use_id: Any, tool_response: Any) -> None:
            return None

        def end_tool_span_with_error(self, tool_use_id: Any, error: Any) -> None:
            return None

        def end_all_in_flight(self) -> None:
            return None

    options = _DummyOptions()
    options.hooks = {"PreToolUse": [_hook_matcher(user_hook)]}
    merged = wrappers._merge_hooks(options, tool_tracker=_NoopTracker())
    assert merged is not None

    async def _run() -> None:
        await _run_hooks(merged, "PreToolUse", {"tool_use_id": "toolu_123"})

    import asyncio

    asyncio.run(_run())
    assert user_called


@pytest.mark.asyncio
async def test_task_subagent_span_parenting(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """Task tool invocations should create a subagent span under the tool span."""
    from opentelemetry import trace as trace_api

    import openinference.instrumentation.claude_agent_sdk._wrappers as wrappers

    trace_api.set_tracer_provider(tracer_provider)
    tracer = tracer_provider.get_tracer(__name__)

    messages = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": "sess-root",
            "model": "claude-test",
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_task_1",
                        "name": "Task",
                        "input": {"objective": "do the thing"},
                    }
                ]
            },
        },
        {
            "type": "system",
            "subtype": "init",
            "session_id": "sess-sub",
            "model": "claude-test",
            "parent_tool_use_id": "toolu_task_1",
        },
        {
            "type": "result",
            "subtype": "success",
            "result": "sub ok",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "total_cost_usd": 0.01,
            "session_id": "sess-sub",
            "parent_tool_use_id": "toolu_task_1",
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_task_1",
                        "content": "sub ok",
                        "is_error": False,
                    }
                ]
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "result": "root ok",
            "usage": {"input_tokens": 2, "output_tokens": 2},
            "total_cost_usd": 0.02,
            "session_id": "sess-root",
        },
    ]

    async def fake_query(*, prompt: str = "", options: Any = None) -> Any:
        for msg in messages:
            yield msg

    wrapper = wrappers._QueryWrapper(tracer)
    async for _ in wrapper(fake_query, None, (), {"prompt": "parent prompt"}):
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    root_span = _span_by_name(spans, "ClaudeAgentSDK.query")
    tool_span = _span_by_name(spans, "Task")
    subagent_span = _span_by_name(spans, "ClaudeAgentSDK.Task")

    assert tool_span.parent is not None
    assert tool_span.parent.span_id == root_span.context.span_id
    assert subagent_span.parent is not None
    assert subagent_span.parent.span_id == tool_span.context.span_id

    root_attrs = dict(root_span.attributes or {})
    assert root_attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == AGENT_KIND
    assert root_attrs.get(SpanAttributes.OUTPUT_VALUE) == "root ok"
    assert root_attrs.get(SpanAttributes.SESSION_ID) == "sess-root"

    subagent_attrs = dict(subagent_span.attributes or {})
    assert subagent_attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == AGENT_KIND
    assert subagent_attrs.get(SpanAttributes.AGENT_NAME) == "Task"
    assert subagent_attrs.get(SpanAttributes.OUTPUT_VALUE) == "sub ok"
    assert subagent_attrs.get(SpanAttributes.SESSION_ID) == "sess-sub"


# ---- Integration tests: cassette-based (always run in CI) ----


@pytest.mark.timeout(30)
@pytest.mark.asyncio
async def test_query_tool_spans_from_messages(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
    cassette_transport: Any,
) -> None:
    """query() creates TOOL spans for tool-use messages."""
    from claude_agent_sdk import ClaudeAgentOptions, query

    options = ClaudeAgentOptions(allowed_tools=["Bash"], permission_mode="bypassPermissions")

    async for _ in query(prompt=_TOOL_PROMPT, options=options, transport=cassette_transport):
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    agent_span = _span_by_name(spans, "ClaudeAgentSDK.query")
    agent_model = dict(agent_span.attributes or {}).get(SpanAttributes.LLM_MODEL_NAME)
    assert isinstance(agent_model, str)
    tool_spans = _tool_spans(spans)
    bash_spans = [span for span in tool_spans if span.name == "Bash"]
    assert bash_spans
    for span in bash_spans:
        assert span.parent is not None
        assert span.parent.span_id == agent_span.context.span_id
        attrs = dict(span.attributes or {})
        assert attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND, None) == TOOL_KIND
        assert attrs.pop(SpanAttributes.TOOL_NAME, None) == "Bash"
        tool_id = attrs.pop(SpanAttributes.TOOL_ID, None)
        assert isinstance(tool_id, str)
        assert tool_id
        tool_params = attrs.pop(SpanAttributes.TOOL_PARAMETERS, None)
        assert isinstance(tool_params, str)
        parsed_params = json.loads(tool_params)
        assert isinstance(parsed_params, dict)
        command = parsed_params.get("command")
        assert isinstance(command, str)
        assert "wc -c" in command
        assert "pyproject.toml" in command
        assert (
            attrs.pop(SpanAttributes.INPUT_MIME_TYPE, None)
            == OpenInferenceMimeTypeValues.JSON.value
        )
        input_value = attrs.pop(SpanAttributes.INPUT_VALUE, None)
        assert isinstance(input_value, str)
        assert (
            attrs.pop(SpanAttributes.OUTPUT_MIME_TYPE, None)
            == OpenInferenceMimeTypeValues.JSON.value
        )
        output_value = attrs.pop(SpanAttributes.OUTPUT_VALUE, None)
        assert isinstance(output_value, str)
        assert not attrs


@pytest.mark.timeout(30)
@pytest.mark.asyncio
async def test_query_task_subagent_spans(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
    cassette_transport: Any,
) -> None:
    """Agent tool usage should create a subagent span under the agent tool span."""
    from claude_agent_sdk import ClaudeAgentOptions, query

    options = ClaudeAgentOptions(
        allowed_tools=["Bash", "Agent", "TaskOutput"],
        permission_mode="bypassPermissions",
    )

    async for _ in query(prompt=_TASK_PROMPT, options=options, transport=cassette_transport):
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    agent_span = _span_by_name(spans, "ClaudeAgentSDK.query")
    agent_model = dict(agent_span.attributes or {}).get(SpanAttributes.LLM_MODEL_NAME)
    assert isinstance(agent_model, str)
    tool_spans = _tool_spans(spans)
    span_by_id = {span.context.span_id: span for span in spans}
    subagent_spans = []
    for span in spans:
        attrs = dict(span.attributes or {})
        if attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) != AGENT_KIND:
            continue
        if not span.name.startswith("ClaudeAgentSDK."):
            continue
        if span.name == "ClaudeAgentSDK.query":
            continue
        parent = span.parent
        if parent is None:
            continue
        parent_span = span_by_id.get(parent.span_id)
        if parent_span is None:
            continue
        parent_attrs = dict(parent_span.attributes or {})
        if parent_attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) != TOOL_KIND:
            continue
        subagent_spans.append((span, parent_span))

    assert subagent_spans
    subagent_span, parent_tool_span = subagent_spans[0]

    assert parent_tool_span.parent is not None
    assert parent_tool_span.parent.span_id == agent_span.context.span_id
    assert subagent_span.parent is not None
    assert subagent_span.parent.span_id == parent_tool_span.context.span_id

    parent_tool_name = dict(parent_tool_span.attributes or {}).get(SpanAttributes.TOOL_NAME)
    assert isinstance(parent_tool_name, str)
    assert parent_tool_name

    bash_spans = [
        span
        for span in tool_spans
        if span.name == "Bash"
        and span.parent is not None
        and span.parent.span_id == subagent_span.context.span_id
    ]
    assert bash_spans

    parent_tool_attrs = dict(parent_tool_span.attributes or {})
    assert parent_tool_attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND, None) == TOOL_KIND
    assert parent_tool_attrs.pop(SpanAttributes.TOOL_NAME, None) == parent_tool_name
    parent_tool_id = parent_tool_attrs.pop(SpanAttributes.TOOL_ID, None)
    assert isinstance(parent_tool_id, str)
    assert parent_tool_id
    parent_tool_params = parent_tool_attrs.pop(SpanAttributes.TOOL_PARAMETERS, None)
    assert isinstance(parent_tool_params, str)
    parsed_params = json.loads(parent_tool_params)
    assert isinstance(parsed_params, dict)
    assert (
        parent_tool_attrs.pop(SpanAttributes.INPUT_MIME_TYPE, None)
        == OpenInferenceMimeTypeValues.JSON.value
    )
    parent_tool_input_value = parent_tool_attrs.pop(SpanAttributes.INPUT_VALUE, None)
    assert isinstance(parent_tool_input_value, str)
    parent_tool_output_value = parent_tool_attrs.pop(SpanAttributes.OUTPUT_VALUE, None)
    assert isinstance(parent_tool_output_value, str)
    parent_tool_output_mime = parent_tool_attrs.pop(SpanAttributes.OUTPUT_MIME_TYPE, None)
    assert parent_tool_output_mime == OpenInferenceMimeTypeValues.JSON.value
    assert not parent_tool_attrs

    subagent_attrs = dict(subagent_span.attributes or {})
    assert subagent_attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND, None) == AGENT_KIND
    assert subagent_attrs.pop(SpanAttributes.AGENT_NAME, None) == parent_tool_name
    subagent_output_value = subagent_attrs.pop(SpanAttributes.OUTPUT_VALUE, None)
    assert subagent_output_value is None
    subagent_output_mime = subagent_attrs.pop(SpanAttributes.OUTPUT_MIME_TYPE, None)
    assert subagent_output_mime is None
    subagent_session_id = subagent_attrs.pop(SpanAttributes.SESSION_ID, None)
    assert subagent_session_id is None
    subagent_model = subagent_attrs.pop(SpanAttributes.LLM_MODEL_NAME, None)
    assert subagent_model == agent_model
    subagent_prompt_tokens = subagent_attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, None)
    assert subagent_prompt_tokens is None
    subagent_completion_tokens = subagent_attrs.pop(
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
        None,
    )
    assert subagent_completion_tokens is None
    subagent_cache_read = subagent_attrs.pop(
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
        None,
    )
    assert subagent_cache_read is None
    subagent_cache_write = subagent_attrs.pop(
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
        None,
    )
    assert subagent_cache_write is None
    subagent_total_tokens = subagent_attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, None)
    assert subagent_total_tokens is None
    subagent_cost_total = subagent_attrs.pop(SpanAttributes.LLM_COST_TOTAL, None)
    assert subagent_cost_total is None
    assert not subagent_attrs


@pytest.mark.timeout(30)
@pytest.mark.asyncio
async def test_query_tool_fallback_when_hooks_unavailable(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
    cassette_transport: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If hooks cannot be injected, message-based tool tracking should work."""
    from claude_agent_sdk import ClaudeAgentOptions, query

    import openinference.instrumentation.claude_agent_sdk._wrappers as wrappers

    monkeypatch.setattr(wrappers, "_merge_hooks", lambda *args, **kwargs: None)

    options = ClaudeAgentOptions(allowed_tools=["Bash"], permission_mode="bypassPermissions")
    async for _ in query(prompt=_TOOL_PROMPT, options=options, transport=cassette_transport):
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    tool_spans = _tool_spans(spans)
    bash_spans = [span for span in tool_spans if span.name == "Bash"]
    assert bash_spans
    for span in bash_spans:
        attrs = dict(span.attributes or {})
        assert attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND, None) == TOOL_KIND
        assert attrs.pop(SpanAttributes.TOOL_NAME, None) == "Bash"
        tool_id = attrs.pop(SpanAttributes.TOOL_ID, None)
        assert isinstance(tool_id, str)
        assert tool_id
        tool_params = attrs.pop(SpanAttributes.TOOL_PARAMETERS, None)
        assert isinstance(tool_params, str)
        parsed_params = json.loads(tool_params)
        assert isinstance(parsed_params, dict)
        command = parsed_params.get("command")
        assert isinstance(command, str)
        assert "wc -c" in command
        assert "pyproject.toml" in command
        assert (
            attrs.pop(SpanAttributes.INPUT_MIME_TYPE, None)
            == OpenInferenceMimeTypeValues.JSON.value
        )
        input_value = attrs.pop(SpanAttributes.INPUT_VALUE, None)
        assert isinstance(input_value, str)
        assert (
            attrs.pop(SpanAttributes.OUTPUT_MIME_TYPE, None)
            == OpenInferenceMimeTypeValues.JSON.value
        )
        output_value = attrs.pop(SpanAttributes.OUTPUT_VALUE, None)
        assert isinstance(output_value, str)
        assert not attrs


@pytest.mark.timeout(30)
@pytest.mark.asyncio
async def test_client_tool_hooks_create_tool_spans(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
    cassette_transport: Any,
) -> None:
    """Client receive_response() creates TOOL spans."""
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    options = ClaudeAgentOptions(allowed_tools=["Bash"], permission_mode="bypassPermissions")
    async with ClaudeSDKClient(options=options, transport=cassette_transport) as client:
        await client.query(_TOOL_PROMPT)
        async for _ in client.receive_response():
            pass

    spans = in_memory_span_exporter.get_finished_spans()
    agent_span = _span_by_name(spans, "ClaudeAgentSDK.ClaudeSDKClient.receive_response")
    tool_spans = _tool_spans(spans)
    bash_spans = [span for span in tool_spans if span.name == "Bash"]
    assert bash_spans
    for span in bash_spans:
        assert span.parent is not None
        assert span.parent.span_id == agent_span.context.span_id
        attrs = dict(span.attributes or {})
        assert attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND, None) == TOOL_KIND
        assert attrs.pop(SpanAttributes.TOOL_NAME, None) == "Bash"
        tool_id = attrs.pop(SpanAttributes.TOOL_ID, None)
        assert isinstance(tool_id, str)
        assert tool_id
        tool_params = attrs.pop(SpanAttributes.TOOL_PARAMETERS, None)
        assert isinstance(tool_params, str)
        parsed_params = json.loads(tool_params)
        assert isinstance(parsed_params, dict)
        command = parsed_params.get("command")
        assert isinstance(command, str)
        assert "wc -c" in command
        assert "pyproject.toml" in command
        assert (
            attrs.pop(SpanAttributes.INPUT_MIME_TYPE, None)
            == OpenInferenceMimeTypeValues.JSON.value
        )
        input_value = attrs.pop(SpanAttributes.INPUT_VALUE, None)
        assert isinstance(input_value, str)
        assert (
            attrs.pop(SpanAttributes.OUTPUT_MIME_TYPE, None)
            == OpenInferenceMimeTypeValues.JSON.value
        )
        output_value = attrs.pop(SpanAttributes.OUTPUT_VALUE, None)
        assert isinstance(output_value, str)
        assert not attrs


@pytest.mark.timeout(30)
@pytest.mark.asyncio
async def test_query_real_agent_span(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
    cassette_transport: Any,
) -> None:
    from claude_agent_sdk import query

    prompt = "Reply with exactly the word: ok"
    async for _ in query(prompt=prompt, transport=cassette_transport):
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "ClaudeAgentSDK.query"
    assert span.status.is_ok
    attrs = dict(span.attributes or {})
    assert attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND, None) == AGENT_KIND
    assert attrs.pop(SpanAttributes.INPUT_VALUE, None) == prompt
    assert attrs.pop(SpanAttributes.INPUT_MIME_TYPE, None) == OpenInferenceMimeTypeValues.TEXT.value
    assert attrs.pop(SpanAttributes.OUTPUT_VALUE, None) == "ok"
    assert (
        attrs.pop(SpanAttributes.OUTPUT_MIME_TYPE, None) == OpenInferenceMimeTypeValues.TEXT.value
    )
    session_id = attrs.pop(SpanAttributes.SESSION_ID, None)
    assert isinstance(session_id, str)
    model_name = attrs.pop(SpanAttributes.LLM_MODEL_NAME, None)
    assert isinstance(model_name, str)
    llm_system = attrs.pop(SpanAttributes.LLM_SYSTEM, None)
    assert llm_system == OpenInferenceLLMSystemValues.ANTHROPIC.value
    prompt_tokens = attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, None)
    completion_tokens = attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, None)
    assert isinstance(prompt_tokens, int)
    assert isinstance(completion_tokens, int)
    cache_read_tokens = attrs.pop(
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
        None,
    )
    cache_write_tokens = attrs.pop(
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
        None,
    )
    assert isinstance(cache_read_tokens, int)
    assert isinstance(cache_write_tokens, int)
    total_tokens = attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, None)
    if total_tokens is not None:
        assert total_tokens == prompt_tokens + completion_tokens
    cost_total = attrs.pop(SpanAttributes.LLM_COST_TOTAL, None)
    assert isinstance(cost_total, (int, float))
    # Output messages — text-only assistant turn
    output_msg_content = attrs.pop(
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}.0", None
    )
    assert isinstance(output_msg_content, str)
    output_msg_role = attrs.pop(
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}", None
    )
    assert output_msg_role == "assistant"
    assert not attrs


@pytest.mark.timeout(30)
@pytest.mark.asyncio
async def test_client_real_agent_span(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
    cassette_transport: Any,
) -> None:
    """ClaudeSDKClient flow produces one AGENT span (cassette-based, always runs in CI)."""
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    prompt = "Reply with exactly the word: ok"
    options = ClaudeAgentOptions(allowed_tools=["Bash"])
    async with ClaudeSDKClient(options=options, transport=cassette_transport) as client:
        await client.query(prompt)
        async for _ in client.receive_response():
            pass

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "ClaudeAgentSDK.ClaudeSDKClient.receive_response"
    assert span.status.is_ok
    attrs = dict(span.attributes or {})
    assert attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND, None) == AGENT_KIND
    assert attrs.pop(SpanAttributes.INPUT_VALUE, None) == prompt
    assert attrs.pop(SpanAttributes.INPUT_MIME_TYPE, None) == OpenInferenceMimeTypeValues.TEXT.value
    assert attrs.pop(SpanAttributes.OUTPUT_VALUE, None) == "ok"
    assert (
        attrs.pop(SpanAttributes.OUTPUT_MIME_TYPE, None) == OpenInferenceMimeTypeValues.TEXT.value
    )
    session_id = attrs.pop(SpanAttributes.SESSION_ID, None)
    assert isinstance(session_id, str)
    model_name = attrs.pop(SpanAttributes.LLM_MODEL_NAME, None)
    assert isinstance(model_name, str)
    llm_system = attrs.pop(SpanAttributes.LLM_SYSTEM, None)
    assert llm_system == OpenInferenceLLMSystemValues.ANTHROPIC.value
    prompt_tokens = attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, None)
    completion_tokens = attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, None)
    assert isinstance(prompt_tokens, int)
    assert isinstance(completion_tokens, int)
    cache_read_tokens = attrs.pop(
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
        None,
    )
    cache_write_tokens = attrs.pop(
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
        None,
    )
    assert isinstance(cache_read_tokens, int)
    assert isinstance(cache_write_tokens, int)
    total_tokens = attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, None)
    if total_tokens is not None:
        assert total_tokens == prompt_tokens + completion_tokens
    cost_total = attrs.pop(SpanAttributes.LLM_COST_TOTAL, None)
    assert isinstance(cost_total, (int, float))
    # Output messages — text-only assistant turn
    output_msg_content = attrs.pop(
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}.0", None
    )
    assert isinstance(output_msg_content, str)
    output_msg_role = attrs.pop(
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}", None
    )
    assert output_msg_role == "assistant"
    assert not attrs
