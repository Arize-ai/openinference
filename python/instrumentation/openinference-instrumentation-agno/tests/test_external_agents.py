"""External adapter execution, using deterministic harness responses (no API calls)."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, AsyncIterator, Iterator

import pytest
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import TraceConfig, suppress_tracing, using_attributes
from openinference.instrumentation.agno import AgnoInstrumentor

pytest.importorskip("agno.agents.base")
from agno.agents.antigravity import AntigravityAgent  # noqa: E402
from agno.agents.base import (  # noqa: E402
    BaseExternalAgent,
    ExternalRunMetricsEvent,
    ExternalRunResult,
)
from agno.agents.claude import ClaudeAgent  # noqa: E402
from agno.agents.codex import CodexAgent  # noqa: E402
from agno.agents.dspy import DSPyAgent  # noqa: E402
from agno.agents.langgraph import LangGraphAgent  # noqa: E402
from agno.db.sqlite import SqliteDb  # noqa: E402
from agno.exceptions import RunCancelledException  # noqa: E402
from agno.metrics import RunMetrics  # noqa: E402
from agno.models.response import ToolExecution  # noqa: E402
from agno.run.agent import (  # noqa: E402
    RunContentEvent,
    RunOutput,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from agno.tracing.exporter import DatabaseSpanExporter  # noqa: E402


@pytest.fixture
def instrumentation(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    import openinference.instrumentation.agno as module

    # These tests exercise external adapters, without importing optional native providers.
    monkeypatch.setattr(module, "find_model_subclasses", lambda: [])
    token = context_api.attach(context_api.Context())
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = AgnoInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    yield provider, exporter, instrumentor
    instrumentor.uninstrument()
    provider.shutdown()
    context_api.detach(token)


def _tool() -> ToolExecution:
    return ToolExecution(tool_call_id="call-1", tool_name="shell", tool_args={"command": "pwd"})


async def _answer(self: Any, input: Any, **kwargs: Any) -> ExternalRunResult:
    assert "_openinference_external_context" not in kwargs
    await asyncio.sleep(0)
    tool = _tool()
    tool.result = "test-directory"
    return ExternalRunResult(
        content="answer",
        tools=[tool],
        metrics=RunMetrics(input_tokens=10, output_tokens=5, total_tokens=15, cost=0.01),
    )


async def _stream(self: Any, input: Any, **kwargs: Any) -> AsyncIterator[Any]:
    assert "_openinference_external_context" not in kwargs
    tool = _tool()
    yield ToolCallStartedEvent(run_id=kwargs["run_id"], tool=tool)
    await asyncio.sleep(0)
    tool.result = "test-directory"
    yield ToolCallCompletedEvent(run_id=kwargs["run_id"], tool=tool)
    yield RunContentEvent(run_id=kwargs["run_id"], content="answer")
    yield ExternalRunMetricsEvent(
        metrics=RunMetrics(input_tokens=10, output_tokens=5, total_tokens=15, cost=0.01)
    )


def _mock_adapter(monkeypatch: pytest.MonkeyPatch, cls: Any) -> Any:
    monkeypatch.setattr(cls, "_arun_adapter", _answer)
    monkeypatch.setattr(cls, "_arun_adapter_stream", _stream)
    return cls(name="Test adapter", id="agent-1")


@pytest.mark.parametrize(
    "cls", [BaseExternalAgent, ClaudeAgent, CodexAgent, LangGraphAgent, DSPyAgent, AntigravityAgent]
)
@pytest.mark.parametrize("mode", ["sync", "async", "sync-stream", "async-stream"])
async def test_external_run_identity_tools_and_parent(
    instrumentation: Any,
    monkeypatch: pytest.MonkeyPatch,
    cls: Any,
    mode: str,
) -> None:
    provider, exporter, _ = instrumentation
    agent = _mock_adapter(monkeypatch, cls)
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span("parent") as parent:
        parent_id = parent.get_span_context().span_id
        kwargs = {"run_id": "run-1", "session_id": "session-1", "user_id": "user-1"}
        if mode == "async":
            result = await agent.arun("prompt", **kwargs)
            assert result.content == "answer"
        elif mode == "sync":
            # A sync invocation on an async thread exercises run_coroutine_sync's worker.
            assert agent.run("prompt", **kwargs).content == "answer"
        elif mode == "sync-stream":
            events = list(agent.run("prompt", stream=True, **kwargs))
            assert events[-1].event == "RunCompleted"
        else:
            events = []
            async for event in agent.arun("prompt", stream=True, **kwargs):
                events.append(event)
                assert trace.get_current_span().get_span_context().span_id == parent_id
            assert events[-1].event == "RunCompleted"
        assert trace.get_current_span().get_span_context().span_id == parent_id

    spans = exporter.get_finished_spans()
    runs = [s for s in spans if (s.attributes or {}).get("openinference.span.kind") == "AGENT"]
    assert len(runs) == 1
    run = runs[0]
    assert run.name == f"{cls.__name__}.run"
    assert run.parent and run.parent.span_id == parent_id
    attrs = dict(run.attributes or {})
    assert attrs["agno.agent.id"] == "agent-1"
    assert attrs["agno.run.id"] == "run-1"
    assert attrs["agno.session.id"] == "session-1"
    assert attrs["agno.user.id"] == "user-1"
    assert attrs["input.value"] == "prompt"
    assert attrs["output.value"] == "answer"
    assert attrs["llm.token_count.total"] == 15
    assert attrs["llm.cost.total"] == 0.01
    assert run.status.status_code == trace.StatusCode.OK
    tools = [s for s in spans if (s.attributes or {}).get("openinference.span.kind") == "TOOL"]
    if "stream" in mode:
        assert len(tools) == 1
        assert tools[0].parent.span_id == run.context.span_id
        assert (tools[0].attributes or {})["output.value"] == "test-directory"
    else:
        assert tools == []  # Completed records provide no execution timestamps.
        assert attrs["llm.output_messages.1.message.content"] == "test-directory"


def test_plain_sync_and_generated_ids(
    instrumentation: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, exporter, _ = instrumentation
    agent = _mock_adapter(monkeypatch, BaseExternalAgent)
    result = agent.run("prompt")
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert attrs["agno.run.id"] == result.run_id
    assert attrs["agno.session.id"] == result.session_id
    assert attrs["session.id"] == result.session_id


@pytest.mark.parametrize("stream", [False, True])
async def test_context_and_masking(
    instrumentation: Any,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    provider, exporter, instrumentor = instrumentation
    instrumentor.uninstrument()
    instrumentor.instrument(
        tracer_provider=provider, config=TraceConfig(hide_inputs=True, hide_outputs=True)
    )
    agent = _mock_adapter(monkeypatch, BaseExternalAgent)
    with using_attributes(
        session_id="context-session",
        user_id="context-user",
        metadata={"tenant": "test"},
        tags=["external"],
    ):
        if stream:
            async for _ in agent.arun("secret-prompt", stream=True):
                pass
        else:
            await agent.arun("secret-prompt")
    spans = exporter.get_finished_spans()
    for span in spans:
        attrs = dict(span.attributes or {})
        assert attrs["session.id"] == "context-session"
        assert attrs["user.id"] == "context-user"
        assert attrs["metadata"] == '{"tenant": "test"}'
        assert attrs["tag.tags"] == ("external",)
        serialized = str(attrs)
        assert "secret-prompt" not in serialized
        assert "test-directory" not in serialized
        assert "pwd" not in serialized


@pytest.mark.parametrize("mode", ["sync", "async", "sync-stream", "async-stream"])
async def test_suppression(
    instrumentation: Any, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    _, exporter, _ = instrumentation
    agent = _mock_adapter(monkeypatch, BaseExternalAgent)
    with suppress_tracing():
        if mode == "sync":
            agent.run("prompt")
        elif mode == "async":
            await agent.arun("prompt")
        elif mode == "sync-stream":
            list(agent.run("prompt", stream=True))
        else:
            async for _ in agent.arun("prompt", stream=True):
                pass
    assert not exporter.get_finished_spans()


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_returned_error_and_cancellation(
    instrumentation: Any,
    monkeypatch: pytest.MonkeyPatch,
    cancel: bool,
    stream: bool,
) -> None:
    _, exporter, _ = instrumentation

    async def fail(self: Any, input: Any, **kwargs: Any) -> Any:
        raise RunCancelledException("cancelled") if cancel else RuntimeError("failed")

    async def fail_stream(self: Any, input: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield ToolCallStartedEvent(run_id=kwargs["run_id"], tool=_tool())
        await fail(self, input, **kwargs)

    monkeypatch.setattr(BaseExternalAgent, "_arun_adapter", fail)
    monkeypatch.setattr(BaseExternalAgent, "_arun_adapter_stream", fail_stream)
    agent: Any = BaseExternalAgent()
    if stream:
        events = [e async for e in agent.arun("prompt", stream=True)]
        assert events[-1].event == ("RunCancelled" if cancel else "RunError")
    else:
        await agent.arun("prompt")
    run = next(
        s
        for s in exporter.get_finished_spans()
        if (s.attributes or {}).get("openinference.span.kind") == "AGENT"
    )
    assert run.attributes["agno.run.status"] == ("cancelled" if cancel else "error")
    assert run.status.status_code == (trace.StatusCode.UNSET if cancel else trace.StatusCode.ERROR)


async def test_early_close_ends_tools_and_restores_context(
    instrumentation: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, exporter, _ = instrumentation
    agent = _mock_adapter(monkeypatch, BaseExternalAgent)
    with provider.get_tracer(__name__).start_as_current_span("parent") as parent:
        iterator = agent.arun("prompt", stream=True)
        await anext(iterator)  # RunStarted
        await anext(iterator)  # ToolCallStarted
        await iterator.aclose()
        assert trace.get_current_span() is parent
    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    assert next(s for s in spans if s.name == "shell").attributes["agno.tool.status"] == "cancelled"


async def test_retries_close_pending_tools(
    instrumentation: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, exporter, _ = instrumentation
    attempts = 0

    async def retry_stream(self: Any, input: Any, **kwargs: Any) -> AsyncIterator[Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            yield ToolCallStartedEvent(run_id=kwargs["run_id"], tool=_tool())
            raise RuntimeError("retry")
        async for event in _stream(self, input, **kwargs):
            yield event

    monkeypatch.setattr(BaseExternalAgent, "_arun_adapter_stream", retry_stream)
    agent: Any = BaseExternalAgent(retries=1, delay_between_retries=0)
    events = [e async for e in agent.arun("prompt", stream=True)]
    assert events[-1].event == "RunCompleted"
    tools = [s for s in exporter.get_finished_spans() if s.name == "shell"]
    assert len(tools) == 2
    assert tools[0].attributes["agno.tool.status"] == "incomplete"
    assert tools[1].status.status_code == trace.StatusCode.OK


async def test_parallel_runs_and_nested_instrumentation(
    instrumentation: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, exporter, _ = instrumentation
    tracer = provider.get_tracer(__name__)

    async def nested(self: Any, input: Any, **kwargs: Any) -> Any:
        with tracer.start_as_current_span(f"child-{input}"):
            return await _answer(self, input, **kwargs)

    monkeypatch.setattr(BaseExternalAgent, "_arun_adapter", nested)
    agent: Any = BaseExternalAgent()
    outputs = await asyncio.gather(agent.arun("a"), agent.arun("b"))
    spans = exporter.get_finished_spans()
    runs = {s.attributes["agno.run.id"]: s for s in spans if s.name == "BaseExternalAgent.run"}
    assert len(runs) == 2
    for output, child_name in zip(outputs, ("child-a", "child-b")):
        child = next(s for s in spans if s.name == child_name)
        assert child.parent.span_id == runs[output.run_id].context.span_id


async def test_database_export_and_background_execution(
    instrumentation: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    provider, _, _ = instrumentation
    db = SqliteDb(db_file=str(tmp_path / "traces.db"))
    provider.add_span_processor(SimpleSpanProcessor(DatabaseSpanExporter(db)))
    agent = _mock_adapter(monkeypatch, BaseExternalAgent)
    agent.db = db
    pending = await agent.arun(
        "prompt",
        background=True,
        run_id="background-run",
        session_id="db-session",
        user_id="db-user",
    )
    for _ in range(100):
        if pending.status.value in ("COMPLETED", "completed"):
            break
        await asyncio.sleep(0.01)
    assert pending.status.value.lower() == "completed"
    record = db.get_trace(run_id="background-run")
    assert record is not None
    assert record.session_id == "db-session"
    assert record.user_id == "db-user"


async def test_uninstrument_and_tool_opt_out(
    instrumentation: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, exporter, instrumentor = instrumentation
    agent = _mock_adapter(monkeypatch, BaseExternalAgent)
    instrumentor.uninstrument()
    await agent.arun("prompt")
    assert not exporter.get_finished_spans()
    instrumentor.instrument(tracer_provider=provider, capture_external_tool_spans=False)
    events = [e async for e in agent.arun("prompt", stream=True, yield_run_output=True)]
    assert isinstance(events[-1], RunOutput)
    assert len(exporter.get_finished_spans()) == 1


@pytest.mark.parametrize("harness", ["claude", "codex"])
@pytest.mark.parametrize("stream", [False, True])
async def test_harness_messages_reach_tracing_database(
    instrumentation: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    harness: str,
    stream: bool,
) -> None:
    """Run the real adapters with synthetic SDK messages/notifications."""
    provider, _, _ = instrumentation
    db = SqliteDb(db_file=str(tmp_path / "harness-traces.db"))
    provider.add_span_processor(SimpleSpanProcessor(DatabaseSpanExporter(db)))
    agent: Any
    if harness == "claude":
        import agno.agents.claude.agent as module

        sdk = SimpleNamespace(
            **{
                name: type(name, (SimpleNamespace,), {})
                for name in (
                    "AssistantMessage",
                    "UserMessage",
                    "ResultMessage",
                    "StreamEvent",
                    "TextBlock",
                    "ToolUseBlock",
                    "ToolResultBlock",
                )
            }
        )
        messages = [
            sdk.AssistantMessage(
                content=[sdk.ToolUseBlock(id="tool-1", name="Bash", input={"command": "pwd"})]
            ),
            sdk.UserMessage(
                content=[
                    sdk.ToolResultBlock(
                        tool_use_id="tool-1", content="test-directory", is_error=False
                    )
                ]
            ),
            sdk.AssistantMessage(content=[sdk.TextBlock(text="answer")]),
        ]

        async def query(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            for message in messages:
                yield message

        monkeypatch.setattr(module, "_sdk", lambda: sdk)
        monkeypatch.setattr(ClaudeAgent, "_aquery", query)
        agent = ClaudeAgent(id="claude-test")
        tool_name = "Bash"
    else:
        import agno.agents.codex.agent as codex_module

        item = SimpleNamespace(
            type="commandExecution",
            id="tool-1",
            command="pwd",
            aggregated_output="test-directory",
            exit_code=0,
        )
        notifications = [
            SimpleNamespace(method="item/started", payload=SimpleNamespace(item=item)),
            SimpleNamespace(method="item/completed", payload=SimpleNamespace(item=item)),
            SimpleNamespace(
                method="item/completed",
                payload=SimpleNamespace(
                    item=SimpleNamespace(type="agentMessage", id="message-1", text="answer")
                ),
            ),
            SimpleNamespace(
                method="turn/completed",
                payload=SimpleNamespace(turn=SimpleNamespace(status="completed")),
            ),
        ]

        async def notification_stream() -> AsyncIterator[Any]:
            for notification in notifications:
                yield notification

        async def turn(*args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(stream=notification_stream)

        async def open_thread(*args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(id="thread-1", turn=turn), False

        @asynccontextmanager
        async def client(*args: Any) -> AsyncIterator[Any]:
            yield SimpleNamespace()

        monkeypatch.setattr(codex_module, "_sdk", lambda: SimpleNamespace())
        monkeypatch.setattr(CodexAgent, "_new_client", client)
        monkeypatch.setattr(CodexAgent, "_open_thread", open_thread)
        agent = CodexAgent(id="codex-test")
        tool_name = "shell"
    if stream:
        events = [
            e
            async for e in agent.arun(
                "prompt", stream=True, run_id="harness-run", session_id="harness-session"
            )
        ]
        assert events[-1].content == "answer"
    else:
        output = await agent.arun("prompt", run_id="harness-run", session_id="harness-session")
        assert output.content == "answer"
    record = db.get_trace(run_id="harness-run")
    assert record is not None
    assert record.agent_id == agent.id
    assert record.session_id == "harness-session"
    spans = db.get_spans(trace_id=record.trace_id)
    assert len(spans) == (2 if stream else 1)
    root = next(s for s in spans if s.parent_span_id is None)
    assert root.attributes["output.value"] == "answer"
    if stream:
        tool = next(s for s in spans if s.name == tool_name)
        assert tool.parent_span_id == root.span_id
        assert tool.attributes["output.value"] == "test-directory"


async def test_instrumentation_failure_does_not_change_result(
    instrumentation: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openinference.instrumentation.agno._external import _RunTrace

    _, exporter, _ = instrumentation
    agent = _mock_adapter(monkeypatch, BaseExternalAgent)

    def broken_metrics(*args: Any) -> None:
        raise ValueError("bad telemetry")

    monkeypatch.setattr(_RunTrace, "_metrics", broken_metrics)
    assert (await agent.arun("prompt")).content == "answer"
    assert len(exporter.get_finished_spans()) == 1


def test_external_instrumentation_is_optional_on_older_agno(
    instrumentation: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    provider, _, instrumentor = instrumentation
    instrumentor.uninstrument()
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "agno.agents.base", None)
        instrumentor.instrument(tracer_provider=provider)
        assert instrumentor._original_external_methods == {}
        instrumentor.uninstrument()


@pytest.mark.skipif(
    "parent_tool_call_id" not in ToolExecution.__dataclass_fields__,
    reason="Requires Agno tool lineage support",
)
@pytest.mark.parametrize("parent_finished_first", [False, True])
async def test_claude_subagent_tree_persisted(
    instrumentation: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    parent_finished_first: bool,
) -> None:
    """SDK parent IDs, including delayed children, survive through the real adapter and DB."""
    import agno.agents.claude.agent as module

    provider, _, _ = instrumentation
    db = SqliteDb(db_file=str(tmp_path / "nested-traces.db"))
    provider.add_span_processor(SimpleSpanProcessor(DatabaseSpanExporter(db)))
    sdk = SimpleNamespace(
        **{
            name: type(name, (SimpleNamespace,), {})
            for name in (
                "AssistantMessage",
                "UserMessage",
                "ResultMessage",
                "StreamEvent",
                "TextBlock",
                "ToolUseBlock",
                "ToolResultBlock",
            )
        }
    )

    def start(call_id: str, name: str, parent: Any = None) -> Any:
        return sdk.AssistantMessage(
            parent_tool_use_id=parent,
            content=[sdk.ToolUseBlock(id=call_id, name=name, input={})],
        )

    def end(call_id: str) -> Any:
        return sdk.UserMessage(
            content=[sdk.ToolResultBlock(tool_use_id=call_id, content="done", is_error=False)]
        )

    messages = [start("a", "Agent"), start("b", "Agent")]
    if parent_finished_first:
        messages.append(end("a"))
    messages.extend(
        [
            start("child-a", "Bash", "a"),
            start("child-b", "Read", "b"),
            start("grandchild", "Read", "child-a"),
            start("sibling", "Bash"),
            start("orphan", "Read", "unobserved-parent"),
            end("child-b"),
            end("grandchild"),
            end("child-a"),
            end("sibling"),
            end("orphan"),
            end("b"),
        ]
    )
    if not parent_finished_first:
        messages.append(end("a"))
    messages.append(sdk.AssistantMessage(content=[sdk.TextBlock(text="done")]))

    async def query(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        for message in messages:
            yield message

    monkeypatch.setattr(module, "_sdk", lambda: sdk)
    monkeypatch.setattr(ClaudeAgent, "_aquery", query)
    agent: Any = ClaudeAgent(id="nested-claude", db=db)
    events = [
        e
        async for e in agent.arun(
            "delegate", stream=True, run_id="nested-run", session_id="nested-session"
        )
    ]
    assert events[-1].content == "done"
    record = db.get_trace(run_id="nested-run")
    assert record is not None
    spans = db.get_spans(trace_id=record.trace_id)
    root = next(s for s in spans if s.parent_span_id is None)
    tools = {s.attributes["tool_call.id"]: s for s in spans if "tool_call.id" in s.attributes}
    assert len(spans) == 8
    for child, parent in (("child-a", "a"), ("child-b", "b"), ("grandchild", "child-a")):
        assert tools[child].parent_span_id == tools[parent].span_id
        assert tools[child].attributes["agno.tool.parent_call_id"] == parent
    for call_id in ("a", "b", "sibling", "orphan"):
        assert tools[call_id].parent_span_id == root.span_id
    assert tools["orphan"].attributes["agno.tool.parent_call_id"] == "unobserved-parent"
    assert all(s.status_code == "OK" for s in spans)
