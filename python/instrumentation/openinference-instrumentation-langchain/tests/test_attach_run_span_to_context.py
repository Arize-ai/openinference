import asyncio
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    cast,
)
from uuid import uuid4

import pytest
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    BaseCallbackHandler,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel, FakeListLLM
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from typing_extensions import TypedDict

from openinference.instrumentation import suppress_tracing, using_session
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.langchain._tracer import _AttachedRunSpan

_inner_tracer = trace_api.ProxyTracerProvider().get_tracer(__name__)


def _record_inner_span(name: str = "inner") -> None:
    """Stands in for a span from another instrumentor, e.g. an httpx request."""
    with _inner_tracer.start_as_current_span(name):
        pass


class _ChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages: List[BaseMessage], *_: Any, **__: Any) -> ChatResult:
        _record_inner_span()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="hi"))])

    async def _agenerate(self, messages: List[BaseMessage], *_: Any, **__: Any) -> ChatResult:
        _record_inner_span()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="hi"))])


class _Retriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        _record_inner_span()
        return [Document(page_content=query)]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        _record_inner_span()
        return [Document(page_content=query)]


@tool
def _sync_tool(x: int) -> int:
    """Returns its input."""
    _record_inner_span()
    return x


@tool
async def _async_tool(x: int) -> int:
    """Returns its input."""
    _record_inner_span()
    return x


@tool
def _failing_tool(x: int) -> int:
    """Always raises."""
    _record_inner_span()
    raise ValueError("boom")


def _lambda(x: int) -> int:
    _record_inner_span()
    return x


async def _alambda(x: int) -> int:
    _record_inner_span()
    return x


def _stream(x: int) -> Iterator[int]:
    _record_inner_span()
    yield x


async def _astream(x: int) -> AsyncIterator[int]:
    _record_inner_span()
    yield x


@pytest.fixture(autouse=True)
def inner_tracer(
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: TracerProvider,
) -> trace_api.Tracer:
    tracer = tracer_provider.get_tracer(__name__)
    monkeypatch.setattr(f"{__name__}._inner_tracer", tracer)
    return tracer


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[None]:
    LangChainInstrumentor().instrument(
        tracer_provider=tracer_provider,
        attach_run_span_to_context=True,
    )
    yield


@pytest.fixture(autouse=True)
def no_detach_or_callback_errors(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    """OpenTelemetry logs, rather than raises, the ValueError from a misused detach token."""
    with caplog.at_level(logging.WARNING):
        yield
    problems = [
        record.getMessage()
        for record in caplog.get_records("call")
        if "Failed to detach context" in record.getMessage() or "callback" in record.getMessage()
    ]
    assert not problems


def _by_name(exporter: InMemorySpanExporter, name: str) -> List[ReadableSpan]:
    return [span for span in exporter.get_finished_spans() if span.name == name]


def _one(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    spans = _by_name(exporter, name)
    assert len(spans) == 1, [span.name for span in exporter.get_finished_spans()]
    return spans[0]


def _span_ids(spans: List[ReadableSpan]) -> Set[int]:
    return {span.context.span_id for span in spans if span.context}


def _assert_child_of(child: ReadableSpan, parent: ReadableSpan) -> None:
    assert parent.context is not None
    assert child.parent is not None, f"{child.name} has no parent"
    assert child.parent.span_id == parent.context.span_id
    assert child.context is not None
    assert child.context.trace_id == parent.context.trace_id


def _assert_outer_is_current(outer: trace_api.Span) -> None:
    current = trace_api.get_current_span().get_span_context()
    assert current.span_id == outer.get_span_context().span_id


def test_default_does_not_attach(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    LangChainInstrumentor().uninstrument()
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    tracer = LangChainInstrumentor()._tracer
    assert tracer is not None
    assert tracer.run_inline is False

    with inner_tracer.start_as_current_span("outer"):
        RunnableLambda(_lambda, name="run").invoke(1)

    outer = _one(in_memory_span_exporter, "outer")
    _assert_child_of(_one(in_memory_span_exporter, "inner"), outer)
    _assert_child_of(_one(in_memory_span_exporter, "run"), outer)


@pytest.mark.parametrize(
    "runnable, name",
    [
        pytest.param(RunnableLambda(_lambda, name="run"), "run", id="chain"),
        pytest.param(_sync_tool, "_sync_tool", id="tool"),
        pytest.param(_Retriever(name="run"), "run", id="retriever"),
        pytest.param(_ChatModel(name="run"), "run", id="chat-model"),
    ],
)
def test_span_inside_sync_run_is_child_of_run(
    runnable: Any,
    name: str,
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    payload = {"x": 1} if name == "_sync_tool" else "1"
    with inner_tracer.start_as_current_span("outer"):
        context_before_run = context_api.get_current()
        runnable.invoke(payload)
        assert context_api.get_current() is context_before_run

    run = _one(in_memory_span_exporter, name)
    _assert_child_of(_one(in_memory_span_exporter, "inner"), run)
    _assert_child_of(run, _one(in_memory_span_exporter, "outer"))


@pytest.mark.parametrize(
    "runnable, name",
    [
        pytest.param(RunnableLambda(_alambda, name="run"), "run", id="chain"),
        pytest.param(RunnableLambda(_lambda, name="run"), "run", id="sync-chain-in-executor"),
        pytest.param(_async_tool, "_async_tool", id="tool"),
        pytest.param(_sync_tool, "_sync_tool", id="sync-tool-in-executor"),
        pytest.param(_Retriever(name="run"), "run", id="retriever"),
        pytest.param(_ChatModel(name="run"), "run", id="chat-model"),
    ],
)
async def test_span_inside_async_run_is_child_of_run(
    runnable: Any,
    name: str,
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    payload = {"x": 1} if name.endswith("_tool") else "1"
    with inner_tracer.start_as_current_span("outer") as outer:
        await runnable.ainvoke(payload)
        # LangChain ends async runs from a task of its own, so the run ended in a copy of this
        # context and this one was never restored.
        _assert_outer_is_current(outer)
        _record_inner_span("after")

    run = _one(in_memory_span_exporter, name)
    _assert_child_of(_one(in_memory_span_exporter, "inner"), run)
    _assert_child_of(run, _one(in_memory_span_exporter, "outer"))
    _assert_child_of(_one(in_memory_span_exporter, "after"), _one(in_memory_span_exporter, "outer"))


def test_nested_runs_each_become_current_in_turn(
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    def tool_runner(x: int) -> int:
        _record_inner_span("before_tool")
        _sync_tool.invoke({"x": x})
        _record_inner_span("after_tool")
        return x

    with inner_tracer.start_as_current_span("outer"):
        RunnableLambda(tool_runner, name="chain").invoke(1)

    chain = _one(in_memory_span_exporter, "chain")
    tool_run = _one(in_memory_span_exporter, "_sync_tool")
    _assert_child_of(tool_run, chain)
    _assert_child_of(_one(in_memory_span_exporter, "inner"), tool_run)
    _assert_child_of(_one(in_memory_span_exporter, "before_tool"), chain)
    _assert_child_of(_one(in_memory_span_exporter, "after_tool"), chain)


def test_threads_of_a_batch_do_not_share_a_parent(
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    with inner_tracer.start_as_current_span("outer") as outer:
        RunnableLambda(_lambda, name="run").batch([1, 2, 3, 4], {"max_concurrency": 4})
        _assert_outer_is_current(outer)

    runs = _span_ids(_by_name(in_memory_span_exporter, "run"))
    parents = [
        span.parent.span_id for span in _by_name(in_memory_span_exporter, "inner") if span.parent
    ]
    assert len(runs) == 4
    assert sorted(parents) == sorted(runs)


def test_branches_of_parallel_run_nest_under_their_own_run(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    def left(x: int) -> int:
        _record_inner_span("left_inner")
        return x

    def right(x: int) -> int:
        _record_inner_span("right_inner")
        return x

    RunnableParallel(
        left=RunnableLambda(left, name="left"),
        right=RunnableLambda(right, name="right"),
    ).invoke(1)

    for name in ("left", "right"):
        _assert_child_of(
            _one(in_memory_span_exporter, f"{name}_inner"),
            _one(in_memory_span_exporter, name),
        )


async def test_concurrent_tasks_do_not_share_a_parent(
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    release = asyncio.Event()
    started = []

    async def wait_then_record(x: int) -> int:
        started.append(x)
        if len(started) == 3:
            release.set()
        await asyncio.wait_for(release.wait(), timeout=5)
        _record_inner_span()
        return x

    runnable = RunnableLambda(wait_then_record, name="run")
    with inner_tracer.start_as_current_span("outer") as outer:
        await asyncio.gather(*(runnable.ainvoke(i) for i in range(3)))
        _assert_outer_is_current(outer)

    runs = _span_ids(_by_name(in_memory_span_exporter, "run"))
    parents = [
        span.parent.span_id for span in _by_name(in_memory_span_exporter, "inner") if span.parent
    ]
    assert len(runs) == 3
    assert sorted(parents) == sorted(runs)


class _GraphState(TypedDict):
    value: int


def _graph(node: Any) -> Any:
    graph = StateGraph(_GraphState)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    return graph.compile()


def test_span_inside_langgraph_node_is_child_of_node(
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    def node(state: _GraphState) -> _GraphState:
        _record_inner_span()
        return state

    with inner_tracer.start_as_current_span("outer") as outer:
        _graph(node).invoke({"value": 1})
        _assert_outer_is_current(outer)

    _assert_child_of(_one(in_memory_span_exporter, "inner"), _one(in_memory_span_exporter, "node"))


async def test_span_inside_async_langgraph_node_is_child_of_node(
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    async def node(state: _GraphState) -> _GraphState:
        _record_inner_span()
        return state

    with inner_tracer.start_as_current_span("outer") as outer:
        await _graph(node).ainvoke({"value": 1})
        _assert_outer_is_current(outer)

    _assert_child_of(_one(in_memory_span_exporter, "inner"), _one(in_memory_span_exporter, "node"))


def test_failed_run_restores_context_and_ends_span(
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    with inner_tracer.start_as_current_span("outer") as outer:
        with pytest.raises(ValueError):
            _failing_tool.invoke({"x": 1})
        _assert_outer_is_current(outer)

    run = _one(in_memory_span_exporter, "_failing_tool")
    assert run.status.status_code is StatusCode.ERROR
    _assert_child_of(_one(in_memory_span_exporter, "inner"), run)


def test_stream_restores_context_when_exhausted_or_closed_early(
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    runnable = RunnableLambda(_stream, name="run")
    with inner_tracer.start_as_current_span("outer") as outer:
        assert list(runnable.stream(1)) == [1]
        _assert_outer_is_current(outer)

        stream = cast(Generator[int, None, None], runnable.stream(1))
        next(stream)
        stream.close()
        _assert_outer_is_current(outer)

    runs = _by_name(in_memory_span_exporter, "run")
    inners = _by_name(in_memory_span_exporter, "inner")
    assert len(runs) == len(inners) == 2
    for run, inner in zip(runs, inners):
        _assert_child_of(inner, run)


async def test_async_stream_restores_context_when_exhausted_or_closed_early(
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    with inner_tracer.start_as_current_span("outer") as outer:
        runnable = RunnableLambda(_astream, name="run")
        assert [chunk async for chunk in runnable.astream(1)] == [1]
        _assert_outer_is_current(outer)

        model = GenericFakeChatModel(messages=iter([AIMessage("a b")]), name="chat")
        stream = cast(AsyncGenerator[Any, None], model.astream("hi"))
        await stream.__anext__()
        await stream.aclose()
        _assert_outer_is_current(outer)

    _assert_child_of(_one(in_memory_span_exporter, "inner"), _one(in_memory_span_exporter, "run"))
    _assert_child_of(_one(in_memory_span_exporter, "chat"), _one(in_memory_span_exporter, "outer"))


async def test_async_stream_ended_by_the_event_loop_leaves_no_stale_parent(
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    with inner_tracer.start_as_current_span("outer") as outer:
        stream = cast(AsyncGenerator[int, None], RunnableLambda(_astream, name="run").astream(1))
        await stream.__anext__()
        # RunnableLambda.astream doesn't forward aclose to its inner generator. The event loop
        # finalizes that generator from a task of its own, so this context never sees the run end.
        await stream.aclose()
        del stream
        while not _by_name(in_memory_span_exporter, "run"):
            gc.collect()
            await asyncio.sleep(0)
        _assert_outer_is_current(outer)
        _record_inner_span("after")

    _assert_child_of(_one(in_memory_span_exporter, "after"), _one(in_memory_span_exporter, "outer"))


def test_run_started_in_one_thread_and_ended_in_another_leaves_no_stale_parent(
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    stream = RunnableLambda(_stream, name="run").stream(1)
    with ThreadPoolExecutor(max_workers=1) as worker:
        # The worker thread keeps one context across submissions and is never revisited
        # by the run, so there is no callback that could clean it up.
        assert worker.submit(next, stream).result() == 1
        assert list(stream) == []
        worker.submit(_record_inner_span, "after").result()
        worker.submit(RunnableLambda(_lambda, name="next_run").invoke, 1).result()

    run = _one(in_memory_span_exporter, "run")
    assert run.end_time is not None
    assert _one(in_memory_span_exporter, "after").parent is None
    next_run = _one(in_memory_span_exporter, "next_run")
    assert next_run.parent is None
    assert next_run.context is not None and run.context is not None
    assert next_run.context.trace_id != run.context.trace_id


async def test_run_started_in_one_task_and_ended_in_another_leaves_no_stale_parent(
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    stream = RunnableLambda(_astream, name="run").astream(1)
    started, ended = asyncio.Event(), asyncio.Event()

    async def start_stream_then_wait() -> None:
        await stream.__anext__()
        started.set()
        await ended.wait()
        _record_inner_span("after")

    with inner_tracer.start_as_current_span("outer"):
        task = asyncio.create_task(start_stream_then_wait())
        await asyncio.wait_for(started.wait(), timeout=5)
        assert [chunk async for chunk in stream] == []
        ended.set()
        await task

    outer = _one(in_memory_span_exporter, "outer")
    _assert_child_of(_one(in_memory_span_exporter, "run"), outer)
    _assert_child_of(_one(in_memory_span_exporter, "after"), outer)


def test_runs_ending_out_of_order_in_one_context_restore_the_outer_span(
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    runnable = RunnableLambda(_stream, name="run")
    with inner_tracer.start_as_current_span("outer") as outer:
        first, second = runnable.stream(1), runnable.stream(2)
        next(first)
        next(second)
        assert list(first) == []
        assert list(second) == []
        _assert_outer_is_current(outer)
        _record_inner_span("after")

    _assert_child_of(_one(in_memory_span_exporter, "after"), _one(in_memory_span_exporter, "outer"))


def test_outer_span_exiting_before_run_ends_does_not_raise(
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    stream = RunnableLambda(_stream, name="run").stream(1)
    with inner_tracer.start_as_current_span("outer"):
        next(stream)
    assert list(stream) == []
    assert not trace_api.get_current_span().get_span_context().is_valid


def test_long_lived_context_does_not_accumulate_ended_runs(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    runnable = RunnableLambda(_stream, name="run")
    with ThreadPoolExecutor(max_workers=1) as worker:
        # Each run starts in the worker thread and ends in this one, so the worker's context is
        # never restored and each run starts on top of the one before it.
        for i in range(3):
            stream = runnable.stream(i)
            assert worker.submit(next, stream).result() == i
            assert list(stream) == []
        worker.submit(_record_inner_span, "after").result()
        current = worker.submit(trace_api.get_current_span).result()

    runs = _by_name(in_memory_span_exporter, "run")
    assert len(runs) == 3
    for span in runs + [_one(in_memory_span_exporter, "after")]:
        assert span.parent is None, span.name
    assert len({span.context.trace_id for span in runs if span.context}) == 3
    # The ended runs are skipped rather than kept as a chain of proxies.
    assert isinstance(current, _AttachedRunSpan)
    assert not isinstance(current.__wrapped__, _AttachedRunSpan)
    assert not current.get_span_context().is_valid


@pytest.mark.parametrize("use_async", [False, True])
async def test_prompts_started_together_are_siblings_and_not_attached(
    use_async: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    prompts: List[List[BaseMessage]] = [[HumanMessage("a")], [HumanMessage("b")]]
    with inner_tracer.start_as_current_span("outer") as outer:
        model = _ChatModel(name="run")
        if use_async:
            await model.agenerate(prompts)
        else:
            model.generate(prompts)
        _assert_outer_is_current(outer)

    outer_span = _one(in_memory_span_exporter, "outer")
    runs = _by_name(in_memory_span_exporter, "run")
    inners = _by_name(in_memory_span_exporter, "inner")
    assert len(runs) == len(inners) == 2
    for span in runs + inners:
        _assert_child_of(span, outer_span)


def test_llm_batch_runs_are_siblings(
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    with inner_tracer.start_as_current_span("outer"):
        FakeListLLM(responses=["a", "b"], name="run").batch(["a", "b"])

    outer = _one(in_memory_span_exporter, "outer")
    runs = _by_name(in_memory_span_exporter, "run")
    assert len(runs) == 2
    for run in runs:
        _assert_child_of(run, outer)


def test_separate_trace_keeps_inner_span_in_the_new_trace(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    LangChainInstrumentor().uninstrument()
    LangChainInstrumentor().instrument(
        tracer_provider=tracer_provider,
        attach_run_span_to_context=True,
        separate_trace_from_runtime_context=True,
    )
    with inner_tracer.start_as_current_span("outer") as outer:
        RunnableLambda(_lambda, name="run").invoke(1)
        _assert_outer_is_current(outer)

    run = _one(in_memory_span_exporter, "run")
    assert run.parent is None
    _assert_child_of(_one(in_memory_span_exporter, "inner"), run)


def test_suppress_tracing_attaches_nothing(
    in_memory_span_exporter: InMemorySpanExporter,
    inner_tracer: trace_api.Tracer,
) -> None:
    with inner_tracer.start_as_current_span("outer") as outer:
        with suppress_tracing():
            suppressed = context_api.get_current()
            RunnableLambda(lambda x: x, name="run").invoke(1)
            assert context_api.get_current() is suppressed
        _assert_outer_is_current(outer)

    assert not _by_name(in_memory_span_exporter, "run")


def test_suppression_starting_mid_run_still_releases_the_context(
    inner_tracer: trace_api.Tracer,
) -> None:
    tracer = LangChainInstrumentor()._tracer
    assert tracer is not None
    run_id = uuid4()
    with inner_tracer.start_as_current_span("outer") as outer:
        tracer.on_chain_start({}, {}, run_id=run_id, name="run")
        assert trace_api.get_current_span().get_span_context() != outer.get_span_context()
        with suppress_tracing():
            tracer.on_chain_end({}, run_id=run_id)
        _assert_outer_is_current(outer)


def test_context_attributes_survive_attach_and_restore(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    observed: List[Optional[str]] = []

    def read_session(x: int) -> int:
        observed.append(context_api.get_value("session.id"))  # type: ignore[arg-type]
        return x

    with using_session("session-1"):
        RunnableLambda(read_session, name="run").invoke(1)
        assert context_api.get_value("session.id") == "session-1"

    assert observed == ["session-1"]
    run = _one(in_memory_span_exporter, "run")
    assert run.attributes is not None
    assert run.attributes["session.id"] == "session-1"


class _EventRecorder(BaseCallbackHandler):
    """A handler that isn't inline, like most third-party handlers."""

    def __init__(self) -> None:
        self.events: List[str] = []

    # on_chat_model_start is left out so that LangChain falls back to on_llm_start for chat models.
    def on_llm_start(self, *_: Any, **__: Any) -> None:
        self.events.append("on_llm_start")

    def on_llm_end(self, *_: Any, **__: Any) -> None:
        self.events.append("on_llm_end")

    def on_chain_start(self, *_: Any, **__: Any) -> None:
        self.events.append("on_chain_start")

    def on_chain_end(self, *_: Any, **__: Any) -> None:
        self.events.append("on_chain_end")

    def on_tool_start(self, *_: Any, **__: Any) -> None:
        self.events.append("on_tool_start")

    def on_tool_end(self, *_: Any, **__: Any) -> None:
        self.events.append("on_tool_end")

    def on_retriever_start(self, *_: Any, **__: Any) -> None:
        self.events.append("on_retriever_start")

    def on_retriever_end(self, *_: Any, **__: Any) -> None:
        self.events.append("on_retriever_end")


class _InlineHandler(BaseCallbackHandler):
    run_inline = True


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    "runnable, payload, expected",
    [
        pytest.param(RunnableLambda(_lambda), 1, ["on_chain_start", "on_chain_end"], id="chain"),
        pytest.param(_sync_tool, {"x": 1}, ["on_tool_start", "on_tool_end"], id="tool"),
        pytest.param(_Retriever(), "1", ["on_retriever_start", "on_retriever_end"], id="retriever"),
        pytest.param(_ChatModel(), "1", ["on_llm_start", "on_llm_end"], id="chat-model"),
    ],
)
async def test_other_handlers_receive_every_event(
    use_async: bool,
    runnable: Any,
    payload: Any,
    expected: List[str],
) -> None:
    recorder = _EventRecorder()
    config: Any = {"callbacks": [recorder]}
    if use_async:
        await runnable.ainvoke(payload, config=config)
    else:
        runnable.invoke(payload, config=config)
    assert recorder.events == expected


def test_other_handlers_receive_llm_start_from_sync_llm() -> None:
    recorder = _EventRecorder()
    FakeListLLM(responses=["a"]).invoke("hi", config={"callbacks": [recorder]})
    assert recorder.events == ["on_llm_start", "on_llm_end"]


async def test_other_handlers_miss_async_llm_start_as_with_any_inline_handler() -> None:
    """
    Once any handler is inline, LangChain's AsyncCallbackManager.on_llm_start only calls the
    inline handlers, so the others don't receive on_llm_start from an async text-completion LLM.
    LangChain's own LangSmith tracer is inline too. This checks that the tracer causes nothing
    beyond that.
    """
    recorder = _EventRecorder()
    await FakeListLLM(responses=["a"]).ainvoke("hi", config={"callbacks": [recorder]})

    # Without the instrumentor, so that the inline handler is the only one that differs.
    LangChainInstrumentor().uninstrument()
    recorder_beside_inline_handler = _EventRecorder()
    await FakeListLLM(responses=["a"]).ainvoke(
        "hi", config={"callbacks": [recorder_beside_inline_handler, _InlineHandler()]}
    )

    assert recorder.events == recorder_beside_inline_handler.events
    assert recorder.events[-1:] == ["on_llm_end"]
