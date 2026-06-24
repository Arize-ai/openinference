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
    MessageContentAttributes,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)

TOOL_KIND = OpenInferenceSpanKindValues.TOOL.value
AGENT_KIND = OpenInferenceSpanKindValues.AGENT.value
SESSION_ID = SpanAttributes.SESSION_ID


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
async def test_propagated_session_id_not_overwritten_by_sdk_session(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """Session ID set via using_session() must not be overwritten by the internal
    Claude CLI session UUID emitted in init/result messages."""
    from opentelemetry import trace as trace_api

    import openinference.instrumentation.claude_agent_sdk._wrappers as wrappers
    from openinference.instrumentation import using_session
    from openinference.semconv.trace import SpanAttributes

    APPLICATION_SESSION_ID = "dedf7759-99ee-46ad-a5fc-3837892a0d78"
    CLI_SESSION_ID = "4e00c355-0cb1-4a44-a7ec-50739f9aabcd"

    trace_api.set_tracer_provider(tracer_provider)
    tracer = tracer_provider.get_tracer(__name__)

    messages = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": CLI_SESSION_ID,
            "model": "claude-test",
        },
        {
            "type": "result",
            "subtype": "success",
            "result": "done",
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "total_cost_usd": 0.01,
            "session_id": CLI_SESSION_ID,
        },
    ]

    async def fake_query(*, prompt: str = "", options: Any = None) -> Any:
        for msg in messages:
            yield msg

    wrapper = wrappers._QueryWrapper(tracer)

    with using_session(APPLICATION_SESSION_ID):
        async for _ in wrapper(fake_query, None, (), {"prompt": "hello"}):
            pass

    spans = in_memory_span_exporter.get_finished_spans()
    agent_span = _span_by_name(spans, "ClaudeAgentSDK.query")
    attrs = dict(agent_span.attributes or {})

    assert attrs.get(SpanAttributes.SESSION_ID) == APPLICATION_SESSION_ID, (
        f"Expected propagated session ID {APPLICATION_SESSION_ID!r} to be preserved, "
        f"but got {attrs.get(SpanAttributes.SESSION_ID)!r}. "
        "The instrumentor is overwriting application session IDs with internal CLI UUIDs."
    )


@pytest.mark.asyncio
async def test_sdk_session_id_set_when_none_propagated(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """When no session ID is propagated via OTel context, the SDK session UUID
    should still be written to the span."""
    from opentelemetry import trace as trace_api

    import openinference.instrumentation.claude_agent_sdk._wrappers as wrappers
    from openinference.semconv.trace import SpanAttributes

    CLI_SESSION_ID = "4e00c355-0cb1-4a44-a7ec-50739f9aabcd"

    trace_api.set_tracer_provider(tracer_provider)
    tracer = tracer_provider.get_tracer(__name__)

    messages = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": CLI_SESSION_ID,
            "model": "claude-test",
        },
        {
            "type": "result",
            "subtype": "success",
            "result": "done",
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "total_cost_usd": 0.01,
            "session_id": CLI_SESSION_ID,
        },
    ]

    async def fake_query(*, prompt: str = "", options: Any = None) -> Any:
        for msg in messages:
            yield msg

    wrapper = wrappers._QueryWrapper(tracer)

    async for _ in wrapper(fake_query, None, (), {"prompt": "hello"}):
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    agent_span = _span_by_name(spans, "ClaudeAgentSDK.query")
    attrs = dict(agent_span.attributes or {})

    assert attrs.get(SpanAttributes.SESSION_ID) == CLI_SESSION_ID, (
        f"Expected SDK session ID {CLI_SESSION_ID!r} to be set when none was propagated, "
        f"but got {attrs.get(SpanAttributes.SESSION_ID)!r}."
    )


@pytest.mark.asyncio
async def test_propagated_session_id_not_overwritten_on_error_result(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """Propagated session ID must be preserved even when the result message is an error."""
    from opentelemetry import trace as trace_api

    import openinference.instrumentation.claude_agent_sdk._wrappers as wrappers
    from openinference.instrumentation import using_session
    from openinference.semconv.trace import SpanAttributes

    APPLICATION_SESSION_ID = "app-session-error-path"
    CLI_SESSION_ID = "cli-session-error-path"

    trace_api.set_tracer_provider(tracer_provider)
    tracer = tracer_provider.get_tracer(__name__)

    messages = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": CLI_SESSION_ID,
            "model": "claude-test",
        },
        {
            "type": "result",
            "subtype": "error:tool",
            "errors": [{"code": "tool_error", "message": "boom"}],
            "usage": {"input_tokens": 2, "output_tokens": 1},
            "total_cost_usd": 0.005,
            "session_id": CLI_SESSION_ID,
        },
    ]

    async def fake_query(*, prompt: str = "", options: Any = None) -> Any:
        for msg in messages:
            yield msg

    wrapper = wrappers._QueryWrapper(tracer)

    with using_session(APPLICATION_SESSION_ID):
        async for _ in wrapper(fake_query, None, (), {"prompt": "hello"}):
            pass

    spans = in_memory_span_exporter.get_finished_spans()
    agent_span = _span_by_name(spans, "ClaudeAgentSDK.query")
    attrs = dict(agent_span.attributes or {})

    assert attrs.get(SpanAttributes.SESSION_ID) == APPLICATION_SESSION_ID, (
        f"Expected propagated session ID {APPLICATION_SESSION_ID!r} on error path, "
        f"but got {attrs.get(SpanAttributes.SESSION_ID)!r}."
    )


@pytest.mark.asyncio
async def test_receive_response_preserves_session_id_set_by_span_processor(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """Span processors can set session.id on span start; SDK session IDs must not clobber it."""
    from opentelemetry.sdk.trace import SpanProcessor

    import openinference.instrumentation.claude_agent_sdk._wrappers as wrappers
    from openinference.semconv.trace import SpanAttributes

    APPLICATION_SESSION_ID = "dedf7759-99ee-46ad-a5fc-3837892a0d78"
    CLI_SESSION_ID = "4e00c355-0cb1-4a44-a7ec-50739f9aabcd"

    class SessionOnStart(SpanProcessor):
        def on_start(self, span: Any, parent_context: Any = None) -> None:
            del parent_context
            span.set_attribute(SpanAttributes.SESSION_ID, APPLICATION_SESSION_ID)

        def on_end(self, span: Any) -> None:
            del span

        def shutdown(self) -> None:
            pass

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            del timeout_millis
            return True

    class Client:
        pass

    tracer_provider.add_span_processor(SessionOnStart())
    tracer = tracer_provider.get_tracer(__name__)

    messages = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": CLI_SESSION_ID,
            "model": "claude-test",
        },
        {
            "type": "result",
            "subtype": "success",
            "result": "done",
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "total_cost_usd": 0.01,
            "session_id": CLI_SESSION_ID,
        },
    ]

    async def fake_receive_response(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        for msg in messages:
            yield msg

    wrapper = wrappers._ClientReceiveResponseWrapper(tracer)

    async for _ in wrapper(fake_receive_response, Client(), (), {}):
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    agent_span = _span_by_name(spans, "ClaudeAgentSDK.ClaudeSDKClient.receive_response")
    attrs = dict(agent_span.attributes or {})

    assert attrs.get(SpanAttributes.SESSION_ID) == APPLICATION_SESSION_ID, (
        f"Expected span-processor session ID {APPLICATION_SESSION_ID!r} to be preserved, "
        f"but got {attrs.get(SpanAttributes.SESSION_ID)!r}."
    )


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


@pytest.mark.asyncio
async def test_tool_error_records_real_content(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """Tool result blocks with is_error=True record the real content, not a hardcoded string."""
    from opentelemetry import trace as trace_api

    import openinference.instrumentation.claude_agent_sdk._wrappers as wrappers

    trace_api.set_tracer_provider(tracer_provider)
    tracer = tracer_provider.get_tracer(__name__)

    real_error_text = "Permission denied: /etc/shadow"
    messages = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": "sess-tool-err",
            "model": "claude-test",
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_err_1",
                        "name": "Bash",
                        "input": {"command": "cat /etc/shadow"},
                    }
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_err_1",
                        "content": [{"type": "text", "text": real_error_text}],
                        "is_error": True,
                    }
                ]
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "result": "I could not complete the task due to a tool error.",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "total_cost_usd": 0.001,
            "session_id": "sess-tool-err",
        },
    ]

    async def fake_query(*, prompt: str = "", options: Any = None) -> Any:
        for msg in messages:
            yield msg

    wrapper = wrappers._QueryWrapper(tracer)
    async for _ in wrapper(fake_query, None, (), {"prompt": "read the shadow file"}):
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    tool_spans = _tool_spans(spans)
    bash_spans = [s for s in tool_spans if s.name == "Bash"]
    assert bash_spans, "Expected a Bash tool span"

    bash_span = bash_spans[0]
    assert bash_span.status.status_code == StatusCode.ERROR

    # The real error content must appear in the status description, not a hardcoded string.
    assert real_error_text in bash_span.status.description, (
        f"Expected real error text in status description, got: {bash_span.status.description!r}"
    )
    assert "Tool execution error" not in bash_span.status.description, (
        "Hardcoded generic string must not appear when real content is available"
    )

    # The real error content must also appear in the recorded exception.
    error_events = [e for e in bash_span.events if e.name == "exception"]
    assert error_events, "Expected an exception event on the tool span"
    exception_msg = error_events[0].attributes.get("exception.message", "")
    assert real_error_text in exception_msg, (
        f"Expected real error in exception.message, got: {exception_msg!r}"
    )

    # The real error content must NOT be set as output attributes.
    attrs = dict(bash_span.attributes or {})
    assert SpanAttributes.OUTPUT_VALUE not in attrs, (
        "output.value must not be set on a failed tool span"
    )
    assert SpanAttributes.OUTPUT_MIME_TYPE not in attrs, (
        "output.mime_type must not be set on a failed tool span"
    )


def test_tool_error_string_remains_unquoted(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """Hook-based string errors should remain human-readable, not JSON-quoted."""
    from opentelemetry import trace as trace_api

    import openinference.instrumentation.claude_agent_sdk._wrappers as wrappers

    trace_api.set_tracer_provider(tracer_provider)
    tracer = tracer_provider.get_tracer(__name__)
    tracker = wrappers._ToolSpanTracker(tracer, None)

    tracker.start_tool_span(
        tool_name="Bash",
        tool_input={"command": "cat /etc/shadow"},
        tool_use_id="toolu_err_2",
    )
    tracker.end_tool_span_with_error("toolu_err_2", "permission denied")

    spans = in_memory_span_exporter.get_finished_spans()
    bash_span = _span_by_name(spans, "Bash")

    assert bash_span.status.status_code == StatusCode.ERROR
    assert bash_span.status.description == "permission denied"

    error_events = [e for e in bash_span.events if e.name == "exception"]
    assert error_events, "Expected an exception event on the tool span"
    assert error_events[0].attributes.get("exception.message") == "permission denied"


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
            self,
            tool_name: Any,
            tool_input: Any,
            tool_use_id: Any,
            parent_tool_use_id: Any = None,
        ) -> None:
            del tool_name, tool_input, tool_use_id, parent_tool_use_id
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


@pytest.mark.asyncio
async def test_thinking_blocks_mixed_sequence_uses_unified_contents_index(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """Mixed output sequence uses a single message.contents index preserving order."""
    from opentelemetry import trace as trace_api

    import openinference.instrumentation.claude_agent_sdk._wrappers as wrappers

    trace_api.set_tracer_provider(tracer_provider)
    tracer = tracer_provider.get_tracer(__name__)

    # Provider Output Sequence: thinking -> tool_use -> redacted_thinking -> text
    messages = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": "sess-thinking",
            "model": "claude-sonnet-4-6",
        },
        {
            "type": "assistant",
            "role": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Let me reason through this carefully.",
                        "signature": "sig_abc123",
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_001",
                        "name": "Bash",
                        "input": {"command": "echo hi"},
                    },
                    {
                        "type": "redacted_thinking",
                        "data": "redacted_blob_xyz",
                    },
                    {
                        "type": "text",
                        "text": "The answer is 42.",
                    },
                ],
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "result": "The answer is 42.",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "total_cost_usd": 0.01,
            "session_id": "sess-thinking",
        },
    ]

    async def fake_query(*, prompt: str = "", options: Any = None) -> Any:
        for msg in messages:
            yield msg

    wrapper = wrappers._QueryWrapper(tracer)
    async for _ in wrapper(fake_query, None, (), {"prompt": "Think and use a tool"}):
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    span = _span_by_name(spans, "ClaudeAgentSDK.query")
    attrs = dict(span.attributes or {})

    msg_prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    contents_prefix = f"{msg_prefix}.{MessageAttributes.MESSAGE_CONTENTS}"

    assert attrs.get(f"{msg_prefix}.{MessageAttributes.MESSAGE_ROLE}") == "assistant"

    # Index 0 has thinking block
    assert (
        attrs.get(f"{contents_prefix}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}")
        == "reasoning"
    )
    assert (
        attrs.get(f"{contents_prefix}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}")
        == "Let me reason through this carefully."
    )
    assert (
        attrs.get(f"{contents_prefix}.0.{MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE}")
        == "sig_abc123"
    )
    assert f"{contents_prefix}.0.{MessageContentAttributes.MESSAGE_CONTENT_ID}" not in attrs

    # Index 1 has tool_use block
    assert (
        attrs.get(f"{contents_prefix}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}")
        == "tool_use"
    )
    assert attrs.get(f"{contents_prefix}.1.{ToolCallAttributes.TOOL_CALL_ID}") == "toolu_001"
    assert attrs.get(f"{contents_prefix}.1.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}") == "Bash"
    tool_args = attrs.get(
        f"{contents_prefix}.1.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    )
    assert isinstance(tool_args, str)
    assert json.loads(tool_args) == {"command": "echo hi"}

    # Index 2 has redacted_thinking block
    assert (
        attrs.get(f"{contents_prefix}.2.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}")
        == "reasoning"
    )
    assert (
        attrs.get(f"{contents_prefix}.2.{MessageContentAttributes.MESSAGE_CONTENT_DATA}")
        == "redacted_blob_xyz"
    )
    assert f"{contents_prefix}.2.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}" not in attrs
    assert f"{contents_prefix}.2.{MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE}" not in attrs
    assert f"{contents_prefix}.2.{MessageContentAttributes.MESSAGE_CONTENT_ID}" not in attrs

    # Index 3 has text block
    assert (
        attrs.get(f"{contents_prefix}.3.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}") == "text"
    )
    assert (
        attrs.get(f"{contents_prefix}.3.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}")
        == "The answer is 42."
    )

    # Legacy keys must NOT be present when thinking blocks are in the message
    assert f"{msg_prefix}.{MessageAttributes.MESSAGE_CONTENT}.0" not in attrs
    assert (
        f"{msg_prefix}.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_ID}"
        not in attrs
    )


@pytest.mark.asyncio
async def test_text_only_message_uses_legacy_content_keys(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """Messages without thinking blocks continue to use legacy message.content keys."""
    from opentelemetry import trace as trace_api

    import openinference.instrumentation.claude_agent_sdk._wrappers as wrappers

    trace_api.set_tracer_provider(tracer_provider)
    tracer = tracer_provider.get_tracer(__name__)

    messages = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": "sess-text",
            "model": "claude-sonnet-4-6",
        },
        {
            "type": "assistant",
            "role": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Just a plain reply."}],
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "result": "Just a plain reply.",
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "total_cost_usd": 0.001,
            "session_id": "sess-text",
        },
    ]

    async def fake_query(*, prompt: str = "", options: Any = None) -> Any:
        for msg in messages:
            yield msg

    wrapper = wrappers._QueryWrapper(tracer)
    async for _ in wrapper(fake_query, None, (), {"prompt": "Say something"}):
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    span = _span_by_name(spans, "ClaudeAgentSDK.query")
    attrs = dict(span.attributes or {})

    msg_prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"

    # Legacy path with no thinking blocks should work
    assert attrs.get(f"{msg_prefix}.{MessageAttributes.MESSAGE_ROLE}") == "assistant"
    assert attrs.get(f"{msg_prefix}.{MessageAttributes.MESSAGE_CONTENT}.0") == "Just a plain reply."

    # message.contents must NOT be present for non-thinking messages
    assert not any(
        k.startswith(f"{msg_prefix}.{MessageAttributes.MESSAGE_CONTENTS}") for k in attrs
    )


@pytest.mark.asyncio
async def test_missing_parent_hook_defers_to_message_parent_for_multiple_agents(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """When hooks omit parent ids, message parent ids remain authoritative."""
    from opentelemetry import trace as trace_api

    import openinference.instrumentation.claude_agent_sdk._wrappers as wrappers

    trace_api.set_tracer_provider(tracer_provider)
    tracer = tracer_provider.get_tracer(__name__)

    root_span = tracer.start_span("root")

    subagent_spans: dict[str, Any] = {}
    tracker_holder: dict[str, Any] = {}

    def _resolve_parent_span(parent_tool_use_id: Any) -> Any:
        key = str(parent_tool_use_id)
        if key not in subagent_spans:
            tracker = tracker_holder["tracker"]
            parent_tool_span = tracker.get_in_flight_span(parent_tool_use_id)
            ctx = trace_api.set_span_in_context(parent_tool_span or root_span)
            subagent_spans[key] = tracer.start_span(
                "ClaudeAgentSDK.Task",
                context=ctx,
            )
        return subagent_spans[key]

    tracker = wrappers._ToolSpanTracker(
        tracer,
        root_span,
        parent_span_resolver=_resolve_parent_span,
    )
    tracker_holder["tracker"] = tracker

    options = _DummyOptions()
    options.hooks = wrappers._create_tool_hook_matchers(tracker)

    # Two root-level subagent tools can be in flight at the same time. Both have
    # no parent in the root assistant message and must remain root-level siblings.
    wrappers._update_tool_spans_from_messages(
        {
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_agent_1",
                        "name": "Agent",
                        "input": {"prompt": "delegate one"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_agent_2",
                        "name": "Agent",
                        "input": {"prompt": "delegate two"},
                    },
                ]
            },
            "parent_tool_use_id": None,
        },
        tracker,
        parent_tool_use_id=None,
    )

    # The hook payload is missing the parent, so it must not create a span by
    # guessing from whichever Agent tool is most recently active.
    await _run_hooks(
        options,
        "PreToolUse",
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "echo from agent one"},
            "tool_use_id": "toolu_bash_1",
            "parent_tool_use_id": None,
        },
    )
    assert tracker.get_in_flight_span("toolu_bash_1") is None

    # The message stream later carries the actual parent id for this Bash call.
    wrappers._update_tool_spans_from_messages(
        {
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_bash_1",
                        "name": "Bash",
                        "input": {"command": "echo from agent one"},
                    },
                ]
            },
            "parent_tool_use_id": "toolu_agent_1",
        },
        tracker,
        parent_tool_use_id="toolu_agent_1",
    )
    tracker.end_tool_span("toolu_bash_1", "alpha")
    tracker.end_tool_span("toolu_agent_1", "agent one")
    tracker.end_tool_span("toolu_agent_2", "agent two")

    for span in subagent_spans.values():
        span.end()
    root_span.end()

    spans = in_memory_span_exporter.get_finished_spans()
    root = _span_by_name(spans, "root")
    agent_spans = [
        s
        for s in spans
        if s.name == "Agent"
        and dict(s.attributes or {}).get(SpanAttributes.TOOL_ID)
        in {"toolu_agent_1", "toolu_agent_2"}
    ]
    assert len(agent_spans) == 2
    agent_by_tool_id = {dict(s.attributes or {})[SpanAttributes.TOOL_ID]: s for s in agent_spans}
    subagent_span = subagent_spans["toolu_agent_1"]
    bash_span = _span_by_name(spans, "Bash")

    for agent_span in agent_spans:
        assert agent_span.parent is not None
        assert agent_span.parent.span_id == root.context.span_id

    assert subagent_span.parent is not None
    assert subagent_span.parent.span_id == agent_by_tool_id["toolu_agent_1"].context.span_id
    assert bash_span.parent is not None
    assert bash_span.parent.span_id == subagent_span.context.span_id
    assert "toolu_agent_2" not in subagent_spans


@pytest.mark.asyncio
async def test_missing_parent_hook_keeps_root_tool_sibling_at_root(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: Any,
) -> None:
    """Root-level message tools stay at root when their hook payload has no parent."""
    from opentelemetry import trace as trace_api

    import openinference.instrumentation.claude_agent_sdk._wrappers as wrappers

    trace_api.set_tracer_provider(tracer_provider)
    tracer = tracer_provider.get_tracer(__name__)

    root_span = tracer.start_span("root")
    subagent_spans: dict[str, Any] = {}

    def _resolve_parent_span(parent_tool_use_id: Any) -> Any:
        key = str(parent_tool_use_id)
        if key not in subagent_spans:
            subagent_spans[key] = tracer.start_span(
                "ClaudeAgentSDK.Agent",
                context=trace_api.set_span_in_context(root_span),
            )
        return subagent_spans[key]

    tracker = wrappers._ToolSpanTracker(
        tracer,
        root_span,
        parent_span_resolver=_resolve_parent_span,
    )

    options = _DummyOptions()
    options.hooks = wrappers._create_tool_hook_matchers(tracker)

    await _run_hooks(
        options,
        "PreToolUse",
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "echo root"},
            "tool_use_id": "toolu_bash_root",
            "parent_tool_use_id": None,
        },
    )
    assert tracker.get_in_flight_span("toolu_bash_root") is None

    wrappers._update_tool_spans_from_messages(
        {
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_agent_1",
                        "name": "Agent",
                        "input": {"prompt": "delegate"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_bash_root",
                        "name": "Bash",
                        "input": {"command": "echo root"},
                    },
                ]
            },
            "parent_tool_use_id": None,
        },
        tracker,
        parent_tool_use_id=None,
    )
    tracker.end_tool_span("toolu_bash_root", "root")
    tracker.end_tool_span("toolu_agent_1", "agent")
    for span in subagent_spans.values():
        span.end()
    root_span.end()

    spans = in_memory_span_exporter.get_finished_spans()
    root = _span_by_name(spans, "root")
    agent_span = _span_by_name(spans, "Agent")
    bash_span = _span_by_name(spans, "Bash")

    assert agent_span.parent is not None
    assert agent_span.parent.span_id == root.context.span_id
    assert bash_span.parent is not None
    assert bash_span.parent.span_id == root.context.span_id
    assert not subagent_spans
