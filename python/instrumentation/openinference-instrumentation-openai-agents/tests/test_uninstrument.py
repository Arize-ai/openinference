"""uninstrument() removes the tracing processor; re-instrumenting mid-run leaves no span open."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Iterator, Optional

import pytest
from agents import (
    Agent,
    Runner,
    add_trace_processor,
    function_tool,
    set_trace_processors,
)
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.tracing import (
    Span,
    Trace,
    TracingProcessor,
    agent_span,
    custom_span,
    get_trace_provider,
    trace,
)
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from opentelemetry.context import Context
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace import Span as SdkSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from openinference.instrumentation.openai_agents._processor import (
    OpenInferenceTracingProcessor,
)


class _Recorder(TracingProcessor):
    """An Agents SDK processor that records which traces it saw start."""

    def __init__(self) -> None:
        self.trace_names: list[str] = []

    def on_trace_start(self, trace: Trace) -> None:
        self.trace_names.append(trace.name)

    def on_trace_end(self, trace: Trace) -> None:
        pass

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


class _OpenSpans(SpanProcessor):
    """Tracks OTel spans that were started but never ended."""

    def __init__(self) -> None:
        self.started: dict[int, SdkSpan] = {}
        self.ended: set[int] = set()

    def on_start(self, span: SdkSpan, parent_context: Optional[Context] = None) -> None:
        self.started[span.context.span_id] = span

    def on_end(self, span: ReadableSpan) -> None:
        assert span.context is not None
        self.ended.add(span.context.span_id)

    def never_ended(self) -> list[str]:
        return sorted(s.name for k, s in self.started.items() if k not in self.ended)


@function_tool
def search_documents(query: str) -> list[dict[str, Any]]:
    """Search the knowledge base."""
    return [{"id": "doc-1", "content": f"About {query}"}]


class _FakeModel(Model):
    """Calls search_documents on the first turn, then answers. When ``gate`` is given,
    the second turn waits on it, which holds the run in flight."""

    def __init__(self, gate: Optional[asyncio.Event] = None) -> None:
        self.calls = 0
        self.gate = gate
        self.waiting = asyncio.Event()

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        output: list[Any]
        if self.calls == 1:
            output = [
                ResponseFunctionToolCall(
                    type="function_call",
                    call_id="call-1",
                    name="search_documents",
                    arguments='{"query":"France"}',
                )
            ]
        else:
            if self.gate is not None:
                self.waiting.set()
                await self.gate.wait()
            output = [
                ResponseOutputMessage(
                    id="m1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[ResponseOutputText(type="output_text", text="Done.", annotations=[])],
                )
            ]
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def _registered() -> tuple[TracingProcessor, ...]:
    return tuple(get_trace_provider()._multi_processor._processors)  # type: ignore[attr-defined]


def _ours() -> list[OpenInferenceTracingProcessor]:
    return [p for p in _registered() if isinstance(p, OpenInferenceTracingProcessor)]


def _run_trace(name: str = "wf") -> None:
    with trace(name):
        with agent_span(name="assistant"):
            with custom_span(name="step"):
                pass


@pytest.fixture(autouse=True)
def _restore_trace_processors() -> Iterator[None]:
    saved = _registered()
    set_trace_processors([])
    yield
    if OpenAIAgentsInstrumentor().is_instrumented_by_opentelemetry:
        OpenAIAgentsInstrumentor().uninstrument()
    set_trace_processors(list(saved))


@pytest.fixture
def open_spans() -> _OpenSpans:
    return _OpenSpans()


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(
    exporter: InMemorySpanExporter, open_spans: _OpenSpans
) -> trace_sdk.TracerProvider:
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(open_spans)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


@pytest.mark.parametrize("exclusive_processor", [True, False])
def test_uninstrument_stops_tracing_and_restores_processors(
    tracer_provider: trace_sdk.TracerProvider,
    exporter: InMemorySpanExporter,
    exclusive_processor: bool,
) -> None:
    other = _Recorder()
    set_trace_processors([other])

    OpenAIAgentsInstrumentor().instrument(
        tracer_provider=tracer_provider, exclusive_processor=exclusive_processor
    )
    _run_trace("traced")
    assert sorted(s.name for s in exporter.get_finished_spans()) == [
        "assistant",
        "step",
        "traced",
    ]

    OpenAIAgentsInstrumentor().uninstrument()
    assert _registered() == (other,)

    exporter.clear()
    _run_trace("untraced")
    assert exporter.get_finished_spans() == ()
    expected = ["untraced"] if exclusive_processor else ["traced", "untraced"]
    assert other.trace_names == expected


@pytest.mark.parametrize("exclusive_processor", [True, False])
def test_uninstrument_keeps_processors_added_after_instrument(
    tracer_provider: trace_sdk.TracerProvider,
    exclusive_processor: bool,
) -> None:
    before = _Recorder()
    set_trace_processors([before])
    OpenAIAgentsInstrumentor().instrument(
        tracer_provider=tracer_provider, exclusive_processor=exclusive_processor
    )
    after = _Recorder()
    add_trace_processor(after)

    OpenAIAgentsInstrumentor().uninstrument()
    assert _registered() == (before, after)


def test_uninstrument_leaves_a_replaced_processor_list_alone(
    tracer_provider: trace_sdk.TracerProvider,
) -> None:
    set_trace_processors([_Recorder()])
    OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
    replacement = _Recorder()
    set_trace_processors([replacement])

    OpenAIAgentsInstrumentor().uninstrument()
    assert _registered() == (replacement,)


@pytest.mark.parametrize("exclusive_processor", [True, False])
def test_reinstrument_during_in_flight_trace_ends_every_span_once(
    tracer_provider: trace_sdk.TracerProvider,
    exporter: InMemorySpanExporter,
    open_spans: _OpenSpans,
    exclusive_processor: bool,
) -> None:
    OpenAIAgentsInstrumentor().instrument(
        tracer_provider=tracer_provider, exclusive_processor=exclusive_processor
    )
    (old,) = _ours()

    in_flight = trace("in flight")
    in_flight.start(mark_as_current=True)
    agent = agent_span(name="assistant")
    agent.start(mark_as_current=True)

    OpenAIAgentsInstrumentor().uninstrument()
    OpenAIAgentsInstrumentor().instrument(
        tracer_provider=tracer_provider, exclusive_processor=exclusive_processor
    )
    (new,) = [p for p in _ours() if p is not old]
    # The old processor stays registered only until the in-flight trace is done.
    assert _registered() == (old, new)

    with custom_span(name="later"):
        pass
    agent.finish(reset_current=True)
    in_flight.finish(reset_current=True)

    assert open_spans.never_ended() == []
    spans = exporter.get_finished_spans()
    assert sorted(s.name for s in spans) == ["assistant", "in flight", "later"]
    assert len({s.context.trace_id for s in spans if s.context}) == 1
    assert _registered() == (new,)


def test_trace_started_after_uninstrument_is_not_traced_while_another_drains(
    tracer_provider: trace_sdk.TracerProvider,
    exporter: InMemorySpanExporter,
    open_spans: _OpenSpans,
) -> None:
    other = _Recorder()
    set_trace_processors([other])
    OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
    (old,) = _ours()

    in_flight = trace("in flight")
    in_flight.start(mark_as_current=True)

    OpenAIAgentsInstrumentor().uninstrument()
    assert _registered() == (other, old)

    with trace("after uninstrument") as later:
        with custom_span(name="untraced step", parent=later):
            pass
    assert exporter.get_finished_spans() == ()
    assert other.trace_names == ["after uninstrument"]

    in_flight.finish(reset_current=True)
    assert [s.name for s in exporter.get_finished_spans()] == ["in flight"]
    assert open_spans.never_ended() == []
    # Drained, so it is no longer registered.
    assert _registered() == (other,)


def test_idle_processor_is_removed_at_uninstrument(
    tracer_provider: trace_sdk.TracerProvider,
) -> None:
    OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
    _run_trace()
    OpenAIAgentsInstrumentor().uninstrument()
    assert _registered() == ()


def test_uninstrument_stops_tracing_when_processors_cannot_be_read(
    tracer_provider: trace_sdk.TracerProvider,
    exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a readable processor list (e.g. a custom TraceProvider) the processor
    cannot be taken out, so it stays registered but traces nothing new."""
    OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
    (processor,) = _ours()
    monkeypatch.setattr(
        "openinference.instrumentation.openai_agents._get_trace_processors", lambda: None
    )
    OpenAIAgentsInstrumentor().uninstrument()
    assert _registered() == (processor,)
    _run_trace()
    assert exporter.get_finished_spans() == ()


def test_uninstrument_stops_tracing_when_processors_cannot_be_replaced(
    tracer_provider: trace_sdk.TracerProvider,
    exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the SDK fails to replace its processor list, the processor stays registered but
    traces nothing new."""
    set_trace_processors([_Recorder()])
    OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
    (processor,) = _ours()

    def fail(processors: list[TracingProcessor]) -> None:
        raise RuntimeError("cannot set processors")

    # Without the SDK's lock the list is replaced through set_trace_processors.
    multi_processor = get_trace_provider()._multi_processor  # type: ignore[attr-defined]
    monkeypatch.setattr(multi_processor, "_lock", None)
    monkeypatch.setattr(multi_processor, "set_processors", fail)
    OpenAIAgentsInstrumentor().uninstrument()
    assert _registered() == (processor,)
    _run_trace()
    assert exporter.get_finished_spans() == ()


def _on_first_read(monkeypatch: pytest.MonkeyPatch, action: Any) -> None:
    """Run `action` right after the instrumentor first reads the SDK's processor list."""
    import openinference.instrumentation.openai_agents as module

    real = module._get_trace_processors
    calls = 0

    def read() -> Optional[list[TracingProcessor]]:
        nonlocal calls
        registered = real()
        calls += 1
        if calls == 1:
            action()
        return registered

    monkeypatch.setattr(module, "_get_trace_processors", read)


@pytest.mark.parametrize("exclusive_processor", [True, False])
def test_processor_added_on_another_thread_during_uninstrument_is_kept(
    tracer_provider: trace_sdk.TracerProvider,
    monkeypatch: pytest.MonkeyPatch,
    exclusive_processor: bool,
) -> None:
    OpenAIAgentsInstrumentor().instrument(
        tracer_provider=tracer_provider, exclusive_processor=exclusive_processor
    )
    added = _Recorder()
    adder = threading.Thread(target=add_trace_processor, args=(added,))

    def add_concurrently() -> None:
        adder.start()
        # Long enough for the add to land unless the list update holds it off.
        adder.join(timeout=0.2)

    _on_first_read(monkeypatch, add_concurrently)
    OpenAIAgentsInstrumentor().uninstrument()
    adder.join()
    assert _registered() == (added,)


def test_processor_added_on_another_thread_during_instrument_is_restored(
    tracer_provider: trace_sdk.TracerProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    added = _Recorder()
    adder = threading.Thread(target=add_trace_processor, args=(added,))

    def add_concurrently() -> None:
        adder.start()
        adder.join(timeout=0.2)

    _on_first_read(monkeypatch, add_concurrently)
    OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
    adder.join()
    monkeypatch.undo()
    OpenAIAgentsInstrumentor().uninstrument()
    assert _registered() == (added,)


def test_trace_started_during_uninstrument_is_not_traced(
    tracer_provider: trace_sdk.TracerProvider,
    exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_trace_processors([_Recorder()])
    OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
    _on_first_read(monkeypatch, lambda: _run_trace("started-during-uninstrument"))
    OpenAIAgentsInstrumentor().uninstrument()
    assert exporter.get_finished_spans() == ()


def test_uninstrument_while_runs_are_in_flight_on_other_threads(
    tracer_provider: trace_sdk.TracerProvider,
    exporter: InMemorySpanExporter,
    open_spans: _OpenSpans,
) -> None:
    OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
    threads = 8
    started = threading.Barrier(threads + 1)
    uninstrumented = threading.Event()

    def run(i: int) -> None:
        with trace(f"run {i}"):
            with agent_span(name="assistant"):
                started.wait()
                uninstrumented.wait()
                with custom_span(name="later"):
                    pass

    workers = [threading.Thread(target=run, args=(i,)) for i in range(threads)]
    for worker in workers:
        worker.start()
    started.wait()
    OpenAIAgentsInstrumentor().uninstrument()
    uninstrumented.set()
    for worker in workers:
        worker.join()

    assert open_spans.never_ended() == []
    assert len(exporter.get_finished_spans()) == threads * 3
    assert _ours() == []


def test_two_processors_draining_at_once_are_both_removed(
    tracer_provider: trace_sdk.TracerProvider,
    exporter: InMemorySpanExporter,
    open_spans: _OpenSpans,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Barrier(2)
    finish = threading.Event()

    def run(name: str) -> None:
        with trace(name):
            started.wait()
            finish.wait()

    # Each cycle leaves its processor draining one run on its own thread.
    workers = []
    for name in ("first", "second"):
        OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
        worker = threading.Thread(target=run, args=(name,))
        worker.start()
        workers.append(worker)
        started.wait()
        OpenAIAgentsInstrumentor().uninstrument()
    assert len(_ours()) == 2

    # Hold each list update until the other thread could also have read the list, so
    # updates that are not serialized would overwrite each other. Without the SDK's lock
    # only our own lock serializes them.
    multi_processor = get_trace_provider()._multi_processor  # type: ignore[attr-defined]
    monkeypatch.setattr(multi_processor, "_lock", None)
    both_read = threading.Barrier(2, timeout=0.5)

    def set_processors_slowly(processors: list[TracingProcessor]) -> None:
        try:
            both_read.wait()
        except threading.BrokenBarrierError:
            pass
        multi_processor._processors = tuple(processors)

    monkeypatch.setattr(multi_processor, "set_processors", set_processors_slowly)
    finish.set()
    for worker in workers:
        worker.join()

    assert _ours() == []
    assert sorted(s.name for s in exporter.get_finished_spans()) == ["first", "second"]
    assert open_spans.never_ended() == []


@pytest.mark.parametrize("exclusive_processor", [True, False])
def test_instrument_uninstrument_cycles(
    tracer_provider: trace_sdk.TracerProvider,
    exporter: InMemorySpanExporter,
    exclusive_processor: bool,
) -> None:
    other = _Recorder()
    set_trace_processors([other])
    for _ in range(3):
        OpenAIAgentsInstrumentor().instrument(
            tracer_provider=tracer_provider, exclusive_processor=exclusive_processor
        )
        assert len(_ours()) == 1
        exporter.clear()
        _run_trace()
        assert sorted(s.name for s in exporter.get_finished_spans()) == ["assistant", "step", "wf"]
        OpenAIAgentsInstrumentor().uninstrument()
        assert _registered() == (other,)


async def test_reinstrument_during_runner_run(
    tracer_provider: trace_sdk.TracerProvider,
    exporter: InMemorySpanExporter,
    open_spans: _OpenSpans,
) -> None:
    OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
    gate = asyncio.Event()
    model = _FakeModel(gate)
    agent = Agent(name="SearchAgent", model=model, tools=[search_documents])
    run = asyncio.create_task(Runner.run(agent, "Tell me about France"))
    await model.waiting.wait()

    # e.g. a configuration reload while a request is being served
    OpenAIAgentsInstrumentor().uninstrument()
    OpenAIAgentsInstrumentor().instrument(
        tracer_provider=tracer_provider,
        tool_span_kinds={"search_documents": "RETRIEVER"},
    )
    gate.set()
    assert (await run).final_output == "Done."

    assert open_spans.never_ended() == []
    first_run = exporter.get_finished_spans()
    ids = [s.context.span_id for s in first_run if s.context]
    assert len(ids) == len(set(ids))
    assert len({s.context.trace_id for s in first_run if s.context}) == 1
    assert len(_ours()) == 1

    # The same run made after re-instrumenting: the in-flight one has the same spans, none
    # missing and none recorded twice, and the new processor carries the new
    # tool_span_kinds.
    exporter.clear()
    agent = Agent(name="SearchAgent", model=_FakeModel(), tools=[search_documents])
    await Runner.run(agent, "Tell me about France")
    second_run = exporter.get_finished_spans()
    assert sorted(s.name for s in first_run) == sorted(s.name for s in second_run)
    assert "search_documents" in [s.name for s in first_run]
    (tool_span,) = [s for s in second_run if s.name == "search_documents"]
    assert dict(tool_span.attributes or {})["openinference.span.kind"] == "RETRIEVER"
    assert open_spans.never_ended() == []
