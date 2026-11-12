import asyncio
import contextvars
import json
import os
from typing import Any, Generator, Iterator, List, Sequence, Tuple, cast

import httpx2
import pytest
import vcr  # type: ignore
from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat
from agno.models.openai.responses import OpenAIResponses
from agno.team import Team
from agno.tools.function import Function
from agno.workflow.parallel import Parallel
from agno.workflow.step import Step
from agno.workflow.types import StepOutput
from agno.workflow.workflow import Workflow
from openai import AsyncOpenAI, OpenAI
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context.contextvars_context import ContextVarsRuntimeContext
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import suppress_tracing
from openinference.instrumentation.agno import AgnoInstrumentor
from openinference.instrumentation.agno.utils import (
    _AGNO_ARUN_SPANNED_CONTEXT_KEY,
    _AGNO_PARENT_NODE_CONTEXT_KEY,
)
from openinference.semconv.trace import SpanAttributes

test_vcr = vcr.VCR(
    serializer="yaml",
    cassette_library_dir="tests/openinference/instrumentation/agno/fixtures/",
    record_mode="none",
    match_on=["uri", "method"],
)


_Isolated = Tuple[TracerProvider, InMemorySpanExporter, ContextVarsRuntimeContext]


@pytest.fixture()
def isolated_instrumentation() -> Iterator[_Isolated]:
    os.environ["OPENAI_API_KEY"] = "fake_key"
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    runtime_context = ContextVarsRuntimeContext()
    AgnoInstrumentor().instrument(tracer_provider=tracer_provider, runtime_context=runtime_context)
    try:
        yield tracer_provider, exporter, runtime_context
    finally:
        AgnoInstrumentor().uninstrument()


_Observed = List[Tuple[trace_api.Span, trace_api.Span]]


def _observe(observed: _Observed, runtime_context: ContextVarsRuntimeContext) -> None:
    observed.append(
        (
            trace_api.get_current_span(),
            trace_api.get_current_span(runtime_context.get_current()),
        )
    )


def _assert_isolated_tree(
    spans: Sequence[ReadableSpan], host_span: trace_api.Span, root_name: str
) -> None:
    host = host_span.get_span_context()
    agno_spans = [span for span in spans if span.context.span_id != host.span_id]
    (root,) = [span for span in agno_spans if span.parent is None]
    assert root.name == root_name
    assert root.context.trace_id != host.trace_id, "agno spans must not join the host trace"
    agno_span_ids = {span.context.span_id for span in agno_spans}
    for span in agno_spans:
        assert span.context.trace_id == root.context.trace_id
        if span is not root:
            assert span.parent is not None and span.parent.span_id in agno_span_ids, (
                f"{span.name} must be parented inside the agno tree"
            )


def test_isolated_agent_run_with_failing_tool_keeps_tree_out_of_host_context(
    isolated_instrumentation: _Isolated,
) -> None:
    tracer_provider, exporter, runtime_context = isolated_instrumentation
    observed: _Observed = []

    def unreliable_search(query: str) -> str:
        """A search tool that fails."""
        _observe(observed, runtime_context)
        raise RuntimeError(f"Search service unavailable for: {query}")

    agent = Agent(
        name="Search Agent",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[Function.from_callable(unreliable_search, name="unreliable_search")],
    )
    with tracer_provider.get_tracer("host").start_as_current_span("host") as host_span:
        with test_vcr.use_cassette(
            "agent_with_failing_tool.yaml", filter_headers=["authorization", "X-API-KEY"]
        ):
            agent.run("Search for Python tutorials", session_id="test_session")
        assert trace_api.get_current_span() is host_span

    spans = exporter.get_finished_spans()
    by_name = {span.name: span for span in spans}
    tool_span = by_name["unreliable_search"]
    assert tool_span.status.status_code == trace_api.StatusCode.ERROR
    assert trace_api.get_current_span(runtime_context.get_current()) is trace_api.INVALID_SPAN, (
        "every span activated in the injected context must be detached after the run"
    )
    assert "OpenAIChat.invoke" in by_name
    ((global_span, isolated_span),) = observed
    assert global_span is host_span, "the run must not replace the host's current span"
    assert isolated_span.get_span_context().span_id == tool_span.context.span_id
    _assert_isolated_tree(spans, host_span, root_name="Search_Agent.run")


def test_isolated_agent_stream_keeps_tree_out_of_host_context(
    isolated_instrumentation: _Isolated,
) -> None:
    tracer_provider, exporter, runtime_context = isolated_instrumentation
    agent = Agent(
        name="Reasoning Agent Stream",
        model=OpenAIResponses(id="o4-mini", reasoning={"effort": "high", "summary": "detailed"}),
        instructions="Use internal reasoning before answering.",
    )
    with tracer_provider.get_tracer("host").start_as_current_span("host") as host_span:
        with test_vcr.use_cassette(
            "agent_run_reasoning_stream.yaml", filter_headers=["authorization", "X-API-KEY"]
        ):
            isolated_span_names = set()
            for _ in agent.run(
                "Count the number of letter 'r' in the word 'strawberry'. Use internal reasoning.",
                session_id="test_session",
                stream=True,
            ):
                assert trace_api.get_current_span() is host_span
                isolated_span = trace_api.get_current_span(runtime_context.get_current())
                isolated_span_names.add(getattr(isolated_span, "name", None))
        assert trace_api.get_current_span() is host_span

    assert "OpenAIResponses.invoke_stream" in isolated_span_names
    spans = exporter.get_finished_spans()
    assert "OpenAIResponses.invoke_stream" in {span.name for span in spans}
    _assert_isolated_tree(spans, host_span, root_name="Reasoning_Agent_Stream.run")


def test_isolated_stream_closed_from_another_context_still_ends_spans(
    isolated_instrumentation: _Isolated,
) -> None:
    """A stream abandoned mid-LLM-call and finalized elsewhere must not raise or drop spans."""
    _, exporter, runtime_context = isolated_instrumentation
    agent = Agent(
        name="Reasoning Agent Stream",
        model=OpenAIResponses(id="o4-mini", reasoning={"effort": "high", "summary": "detailed"}),
        instructions="Use internal reasoning before answering.",
    )
    with test_vcr.use_cassette(
        "agent_run_reasoning_stream.yaml", filter_headers=["authorization", "X-API-KEY"]
    ):
        stream = agent.run(
            "Count the number of letter 'r' in the word 'strawberry'. Use internal reasoning.",
            session_id="test_session",
            stream=True,
        )
        for _ in stream:
            isolated_span = trace_api.get_current_span(runtime_context.get_current())
            if getattr(isolated_span, "name", None) == "OpenAIResponses.invoke_stream":
                break
        else:
            pytest.fail("the stream never entered the LLM span")
        contextvars.copy_context().run(cast(Generator[Any, None, None], stream).close)

    finished = {span.name for span in exporter.get_finished_spans()}
    assert {"OpenAIResponses.invoke_stream", "Reasoning_Agent_Stream.run"} <= finished


async def test_isolated_workflow_arun_keeps_its_marker_out_of_the_global_context(
    isolated_instrumentation: _Isolated,
) -> None:
    _, exporter, runtime_context = isolated_instrumentation
    observed = []

    async def step(step_input: Any) -> StepOutput:
        observed.append(
            (
                context_api.get_value(_AGNO_ARUN_SPANNED_CONTEXT_KEY),
                runtime_context.get_current().get(_AGNO_ARUN_SPANNED_CONTEXT_KEY),
            )
        )
        return StepOutput(step_name="fn_step", content="ok", success=True)

    workflow = Workflow(name="IsolatedWorkflow", steps=[Step(name="fn_step", executor=step)])
    await workflow.arun(input="hello")

    ((global_marker, isolated_marker),) = observed
    assert global_marker is None, "agno-private keys must not leak into the host context"
    assert isolated_marker is not None
    assert sorted(span.name for span in exporter.get_finished_spans()) == [
        "IsolatedWorkflow.arun",
        "fn_step.aexecute",
    ], "the arun marker must still prevent a duplicate _aexecute span"


def test_isolated_workflow_with_parallel_steps_keeps_tree_out_of_host_context(
    isolated_instrumentation: _Isolated,
) -> None:
    tracer_provider, exporter, runtime_context = isolated_instrumentation
    observed: _Observed = []

    def make_step(name: str) -> Step:
        def executor(step_input: Any) -> StepOutput:
            _observe(observed, runtime_context)
            return StepOutput(step_name=name, content="ok", success=True)

        return Step(name=name, executor=executor)

    parallel_steps: Any = [make_step("left"), make_step("right")]
    workflow = Workflow(
        name="IsolatedParallelWorkflow",
        steps=[make_step("first"), Parallel(*parallel_steps, name="fan_out")],
    )
    with tracer_provider.get_tracer("host").start_as_current_span("host") as host_span:
        workflow.run(input="hello")
        assert trace_api.get_current_span() is host_span

    spans = exporter.get_finished_spans()
    span_ids = {span.context.span_id for span in spans}
    assert len(observed) == 3
    for global_span, isolated_span in observed:
        # Parallel steps run in worker threads, whose global context starts empty.
        assert global_span in (host_span, trace_api.INVALID_SPAN), (
            "agno spans must never become current in the global context"
        )
        assert isolated_span.get_span_context().span_id in span_ids
    _assert_isolated_tree(spans, host_span, root_name="IsolatedParallelWorkflow.run")
    by_name = {span.name: span for span in spans}
    assert (by_name["first.execute"].attributes or {})[SpanAttributes.GRAPH_NODE_PARENT_ID] == (
        by_name["IsolatedParallelWorkflow.run"].attributes or {}
    )[SpanAttributes.GRAPH_NODE_ID], "parent node ids must flow through the injected context"


async def test_isolated_background_workflow_keeps_tree_out_of_host_context(
    isolated_instrumentation: _Isolated,
) -> None:
    tracer_provider, exporter, runtime_context = isolated_instrumentation
    observed: _Observed = []

    async def step(step_input: Any) -> StepOutput:
        _observe(observed, runtime_context)
        return StepOutput(step_name="bg_step", content="ok", success=True)

    workflow = Workflow(
        name="IsolatedBackgroundWorkflow", steps=[Step(name="bg_step", executor=step)]
    )
    with tracer_provider.get_tracer("host").start_as_current_span("host") as host_span:
        placeholder = await workflow.arun(input="hello", run_id="RUN-ISOLATED-BG", background=True)
        (task,) = [
            task
            for task in asyncio.all_tasks()
            if task.get_name() == f"workflow-background-{placeholder.run_id}"
        ]
        await asyncio.wait_for(task, timeout=5)

    ((global_span, isolated_span),) = observed
    spans = exporter.get_finished_spans()
    assert global_span is host_span
    assert isolated_span.get_span_context().span_id in {span.context.span_id for span in spans}
    _assert_isolated_tree(spans, host_span, root_name="IsolatedBackgroundWorkflow._aexecute")


def test_suppress_tracing_still_applies_in_isolated_mode(
    isolated_instrumentation: _Isolated,
) -> None:
    _, exporter, _ = isolated_instrumentation

    def step(step_input: Any) -> StepOutput:
        return StepOutput(step_name="fn_step", content="ok", success=True)

    workflow = Workflow(name="SuppressedWorkflow", steps=[Step(name="fn_step", executor=step)])
    with suppress_tracing():
        workflow.run(input="hello")

    assert exporter.get_finished_spans() == ()


def test_uninstrument_during_an_isolated_stream_leaves_no_stale_span(
    isolated_instrumentation: _Isolated,
) -> None:
    """In-flight wrappers must detach from the context they attached to."""
    _, exporter, runtime_context = isolated_instrumentation
    agent = Agent(
        name="Reasoning Agent Stream",
        model=OpenAIResponses(id="o4-mini", reasoning={"effort": "high", "summary": "detailed"}),
        instructions="Use internal reasoning before answering.",
    )
    with test_vcr.use_cassette(
        "agent_run_reasoning_stream.yaml", filter_headers=["authorization", "X-API-KEY"]
    ):
        stream = agent.run(
            "Count the number of letter 'r' in the word 'strawberry'. Use internal reasoning.",
            session_id="test_session",
            stream=True,
        )
        for _ in stream:
            isolated_span = trace_api.get_current_span(runtime_context.get_current())
            if getattr(isolated_span, "name", None) == "OpenAIResponses.invoke_stream":
                break
        else:
            pytest.fail("the stream never entered the LLM span")
        AgnoInstrumentor().uninstrument()
        for _ in stream:
            pass

    assert trace_api.get_current_span(runtime_context.get_current()) is trace_api.INVALID_SPAN
    finished = {span.name for span in exporter.get_finished_spans()}
    assert {"OpenAIResponses.invoke_stream", "Reasoning_Agent_Stream.run"} <= finished


def _delegating_chat_completion(request: httpx2.Request) -> httpx2.Response:
    body = json.loads(request.content)
    tools = {tool["function"]["name"] for tool in body.get("tools") or []}
    answered = any(message.get("role") == "tool" for message in body["messages"])
    message: dict[str, Any] = {"role": "assistant", "content": "done"}
    for name, arguments in (
        ("delegate_task_to_member", {"member_id": "helper", "task": "weather in Paris"}),
        ("get_weather", {"city": "Paris"}),
    ):
        if name in tools and not answered:
            call = {"name": name, "arguments": json.dumps(arguments)}
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": call}],
            }
            break
    choice = {
        "index": 0,
        "message": message,
        "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
    }
    return httpx2.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [choice],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
    )


def _offline_chat_model(is_async: bool) -> OpenAIChat:
    transport = httpx2.MockTransport(_delegating_chat_completion)
    if is_async:
        return OpenAIChat(
            id="gpt-4o-mini",
            async_client=AsyncOpenAI(
                api_key="fake_key",
                base_url="http://mock.invalid/v1",
                http_client=httpx2.AsyncClient(transport=transport),
            ),
        )
    return OpenAIChat(
        id="gpt-4o-mini",
        client=OpenAI(
            api_key="fake_key",
            base_url="http://mock.invalid/v1",
            http_client=httpx2.Client(transport=transport),
        ),
    )


@pytest.mark.parametrize("is_async", [False, True], ids=["run", "arun"])
def test_isolated_team_hands_its_node_id_to_members_through_the_injected_context(
    isolated_instrumentation: _Isolated, is_async: bool
) -> None:
    tracer_provider, exporter, runtime_context = isolated_instrumentation
    observed: List[Tuple[object, object]] = []

    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        observed.append(
            (
                context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY),
                runtime_context.get_current().get(_AGNO_PARENT_NODE_CONTEXT_KEY),
            )
        )
        return f"sunny in {city}"

    member = Agent(
        id="helper", name="Helper", model=_offline_chat_model(is_async), tools=[get_weather]
    )
    team = Team(name="Crew", model=_offline_chat_model(is_async), members=[member])
    with tracer_provider.get_tracer("host").start_as_current_span("host") as host_span:
        if is_async:
            asyncio.run(team.arun("weather?"))
        else:
            team.run("weather?")

    spans = exporter.get_finished_spans()
    (team_span,) = [span for span in spans if span.name.startswith("Crew.")]
    (member_span,) = [span for span in spans if span.name.startswith("Helper.")]
    team_node_id = (team_span.attributes or {})[SpanAttributes.GRAPH_NODE_ID]
    assert (member_span.attributes or {})[SpanAttributes.GRAPH_NODE_PARENT_ID] == team_node_id
    ((global_parent, isolated_parent),) = observed
    assert global_parent is None, "the team handoff key must not leak into the host context"
    assert isolated_parent == team_node_id
    _assert_isolated_tree(spans, host_span, root_name=team_span.name)


def test_isolated_failing_model_call_records_the_error_and_detaches(
    isolated_instrumentation: _Isolated,
) -> None:
    tracer_provider, exporter, runtime_context = isolated_instrumentation

    def reject(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, json={"error": {"message": "bad request", "type": "invalid"}})

    model = OpenAIChat(
        id="gpt-4o-mini",
        client=OpenAI(
            api_key="fake_key",
            base_url="http://mock.invalid/v1",
            http_client=httpx2.Client(transport=httpx2.MockTransport(reject)),
            max_retries=0,
        ),
    )
    with tracer_provider.get_tracer("host").start_as_current_span("host") as host_span:
        try:
            Agent(name="Failing Agent", model=model).run("hi")
        except Exception:
            pass
        assert trace_api.get_current_span() is host_span

    (llm_span,) = [span for span in exporter.get_finished_spans() if span.name.endswith(".invoke")]
    assert llm_span.status.status_code == trace_api.StatusCode.ERROR
    assert [event.name for event in llm_span.events] == ["exception"]
    assert trace_api.get_current_span(runtime_context.get_current()) is trace_api.INVALID_SPAN
