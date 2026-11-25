"""LangSmith-only middleware runs must not detach callback spans from their ancestors."""

from typing import Any, Awaitable, Callable, Optional, Sequence
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langsmith import Client, traceable, tracing_context
from langsmith.run_trees import RunTree
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import suppress_tracing
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.langchain._tracer import OpenInferenceTracer
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes


class _ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "_ToolCallingModel":
        return self


class _PassThrough(AgentMiddleware[Any, Any, Any]):
    def wrap_model_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        return handler(request)

    async def awrap_model_call(self, request: Any, handler: Callable[..., Awaitable[Any]]) -> Any:
        return await handler(request)

    def wrap_tool_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        return handler(request)

    async def awrap_tool_call(self, request: Any, handler: Callable[..., Awaitable[Any]]) -> Any:
        return await handler(request)


class _OuterPassThrough(_PassThrough):
    pass


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("tracing_enabled", [False, True])
@pytest.mark.parametrize("separate_trace", [False, True])
@pytest.mark.parametrize("nested_middleware", [False, True])
async def test_agent_middleware_parents(
    use_async: bool,
    tracing_enabled: bool,
    separate_trace: bool,
    nested_middleware: bool,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    LangChainInstrumentor().uninstrument()
    LangChainInstrumentor().instrument(
        tracer_provider=tracer_provider,
        separate_trace_from_runtime_context=separate_trace,
    )
    middleware = [_OuterPassThrough(), _PassThrough()] if nested_middleware else [_PassThrough()]
    subagent = create_agent(
        model=FakeListChatModel(responses=["hello"]),
        tools=[],
        middleware=middleware,
        name="subagent",
    )

    @traceable
    def delegate() -> str:
        """Delegate to a nested agent through another LangSmith-only run."""
        subagent.invoke({"messages": [HumanMessage(content="hi")]})
        return "hello"

    @traceable
    async def adelegate() -> str:
        """Delegate asynchronously to a nested agent."""
        await subagent.ainvoke({"messages": [HumanMessage(content="hi")]})
        return "hello"

    agent = create_agent(
        model=_ToolCallingModel(
            responses=[
                AIMessage(content="", tool_calls=[{"name": "delegate", "args": {}, "id": "1"}]),
                AIMessage(content="done"),
            ]
        ),
        tools=[StructuredTool.from_function(delegate, coroutine=adelegate)],
        middleware=middleware,
    )
    # Exercise real LangSmith run trees and callbacks without any LangSmith HTTP requests.
    with tracing_context(enabled=tracing_enabled, client=Mock(spec=Client)):
        with tracer_provider.get_tracer("app").start_as_current_span("request"):
            if use_async:
                result = await agent.ainvoke({"messages": [HumanMessage(content="hi")]})
            else:
                result = agent.invoke({"messages": [HumanMessage(content="hi")]})
    assert result["messages"][-1].content == "done"
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 11
    by_id = {span.context.span_id: span for span in spans if span.context}
    graph = next(span for span in spans if span.name == "LangGraph")
    assert graph.context
    for span in spans:
        assert span.context
        if span.name == "request":
            assert span.parent is None
            continue
        assert span.context.trace_id == graph.context.trace_id
        if span.name == "LangGraph" and separate_trace:
            assert span.parent is None
            continue
        assert span.parent
        parent = by_id[span.parent.span_id]
        if span.name == "LangGraph":
            assert parent.name == "request"
        elif span.name in ("_ToolCallingModel", "FakeListChatModel"):
            assert parent.name == "model"
            assert span.attributes
            assert span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == (
                OpenInferenceSpanKindValues.LLM.value
            )
        elif span.name == "delegate":
            assert parent.name == "tools"
            assert span.attributes
            assert span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == (
                OpenInferenceSpanKindValues.TOOL.value
            )
        elif span.name == "subagent":
            assert parent.name == "delegate"
        elif span.name in ("model", "tools"):
            assert parent.name in ("LangGraph", "subagent")
        else:
            pytest.fail(f"Unexpected span: {span.name}")
    tracer = LangChainInstrumentor()._tracer
    assert tracer is not None
    assert not tracer._spans_by_run
    assert not tracer.run_map


@pytest.mark.parametrize("use_async", [False, True])
async def test_middleware_respects_suppression(
    use_async: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup = Mock()
    monkeypatch.setattr(
        "openinference.instrumentation.langchain._tracer.get_current_run_tree", lookup
    )
    agent = create_agent(
        model=FakeListChatModel(responses=["hello"]), tools=[], middleware=[_PassThrough()]
    )
    with tracing_context(enabled=True, client=Mock(spec=Client)), suppress_tracing():
        if use_async:
            await agent.ainvoke({"messages": [HumanMessage(content="hi")]})
        else:
            agent.invoke({"messages": [HumanMessage(content="hi")]})
    assert not in_memory_span_exporter.get_finished_spans()
    lookup.assert_not_called()
    tracer = LangChainInstrumentor()._tracer
    assert tracer is not None
    assert not tracer.run_map
    assert not tracer._spans_by_run


@pytest.mark.parametrize("separate_trace", [False, True])
@pytest.mark.parametrize(
    "scenario",
    ["direct", "detached", "descendant", "unrelated", "no_tree", "root", "malformed", "error"],
)
def test_parent_resolution(
    scenario: str,
    separate_trace: bool,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = OpenInferenceTracer(tracer_provider.get_tracer(__name__), separate_trace)
    root = RunTree(name="root", ls_client=Mock(spec=Client))
    nearest = root.create_child(name="nearest")
    wrapper = nearest.create_child(name="wrapper").create_child(name="inner_wrapper")
    descendant = wrapper.create_child(name="descendant")
    tracer.on_chain_start({}, {}, run_id=root.id, name="root")
    tracer.on_chain_start({}, {}, run_id=nearest.id, parent_run_id=root.id, name="nearest")
    tracer.on_chain_start({}, {}, run_id=descendant.id, parent_run_id=nearest.id, name="descendant")
    parent_id: Optional[UUID] = wrapper.id
    tree = wrapper
    if scenario == "direct":
        parent_id = nearest.id
        tree = RunTree(name="unrelated", ls_client=Mock(spec=Client))
    elif scenario == "detached":
        # LangSmith reconstructs trees from dotted order across callback boundaries.
        tree = RunTree.from_dotted_order(wrapper.dotted_order, client=Mock(spec=Client))
        assert tree.parent_run is None
    elif scenario == "descendant":
        tree = descendant
    elif scenario == "unrelated":
        parent_id = uuid4()
    elif scenario == "root":
        parent_id = None
    elif scenario == "malformed":
        tree.dotted_order = "invalid"
    elif scenario == "error":
        monkeypatch.setattr(
            "openinference.instrumentation.langchain._tracer.get_current_run_tree",
            Mock(side_effect=RuntimeError("unavailable context")),
        )
    child_id = uuid4()
    with tracing_context(parent=tree if scenario != "no_tree" else None, enabled=False):
        with tracer_provider.get_tracer("app").start_as_current_span("request") as request:
            run = tracer.on_chain_start(
                {}, {}, run_id=child_id, parent_run_id=parent_id, name="child"
            )
            assert run.parent_run_id == parent_id
            tracer.on_chain_end({}, run_id=child_id)
    child = next(
        span for span in in_memory_span_exporter.get_finished_spans() if span.name == "child"
    )
    if scenario in ("direct", "detached", "descendant"):
        expected = tracer.get_span(nearest.id)
        assert expected is not None
        assert child.parent == expected.get_span_context()
    elif separate_trace:
        assert child.parent is None
    else:
        assert child.parent == request.get_span_context()
    for run_id in (descendant.id, nearest.id, root.id):
        tracer.on_chain_end({}, run_id=run_id)
    assert not tracer._spans_by_run
    assert not tracer.run_map
