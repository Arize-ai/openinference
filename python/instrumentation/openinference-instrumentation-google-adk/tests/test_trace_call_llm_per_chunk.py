import gc
import threading
from types import SimpleNamespace
from typing import Any, Optional, cast

import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import SpanLimits
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.google_adk import _wrappers
from openinference.instrumentation.google_adk._wrappers import _TraceCallLlm
from openinference.semconv.trace import SpanAttributes


class _CountingTool(FunctionTool):
    """Counts how often its declaration is requested."""

    calls = 0

    def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
        type(self).calls += 1
        return super()._get_declaration()


def _tool() -> _CountingTool:
    def search(query: str) -> dict[str, Any]:
        """A search tool.

        Args:
            query: The search query.
        """
        return {}

    _CountingTool.calls = 0
    return _CountingTool(func=search)


def _request(tool: FunctionTool, history: int = 1) -> LlmRequest:
    return LlmRequest(
        model="gemini-2.0-flash",
        contents=[
            types.Content(role="user" if i % 2 == 0 else "model", parts=[types.Part(text=f"m{i}")])
            for i in range(history)
        ],
        config=types.GenerateContentConfig(system_instruction="be brief"),
        tools_dict={tool.name: tool},
    )


def _chunk(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        partial=True,
    )


def _noop_trace_call_llm(
    invocation_context: Any,
    event_id: str,
    llm_request: Any,
    llm_response: Any,
    span: Any = None,
) -> None:
    return None


def _oi_tracer(tracer_provider: trace_api.TracerProvider) -> trace_api.Tracer:
    """The instrumentor installs an `OITracer`, so spans here are `OpenInferenceSpan`
    proxies that the instrumentor installs at runtime."""
    return cast(trace_api.Tracer, OITracer(tracer_provider.get_tracer(__name__), TraceConfig()))


def test_request_attributes_are_written_once_per_span(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """ADK calls `trace_call_llm` once per streamed chunk against a single span, so the
    request-side attributes must be derived once rather than once per chunk."""
    tool = _tool()
    request = _request(tool)
    tracer = _oi_tracer(tracer_provider)
    wrapped = _TraceCallLlm(tracer)(_noop_trace_call_llm)

    with tracer.start_as_current_span("call_llm"):
        for i in range(5):
            wrapped(None, "e1", request, _chunk(f"c{i}"), None)

    assert tool.calls == 1

    span = in_memory_span_exporter.get_finished_spans()[0]
    attributes = dict(span.attributes or {})
    assert attributes[SpanAttributes.LLM_MODEL_NAME] == "gemini-2.0-flash"
    assert SpanAttributes.INPUT_VALUE in attributes
    assert f"{SpanAttributes.LLM_TOOLS}.0.tool.json_schema" in attributes


def test_response_attributes_are_written_for_every_chunk(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Response-side attributes do change per chunk and must keep being written."""
    tool = _tool()
    request = _request(tool)
    tracer = _oi_tracer(tracer_provider)
    wrapped = _TraceCallLlm(tracer)(_noop_trace_call_llm)

    with tracer.start_as_current_span("call_llm"):
        wrapped(None, "e1", request, _chunk("first"), None)
        wrapped(None, "e1", request, _chunk("last"), None)

    span = in_memory_span_exporter.get_finished_spans()[0]
    assert "last" in str(dict(span.attributes or {})[SpanAttributes.OUTPUT_VALUE])


def test_each_span_gets_its_own_request_attributes(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """The guard is per span: a second `call_llm` span must still be populated."""
    tool = _tool()
    tracer = _oi_tracer(tracer_provider)
    wrapped = _TraceCallLlm(tracer)(_noop_trace_call_llm)

    for model in ("gemini-2.0-flash", "gemini-2.5-flash"):
        request = _request(tool)
        request.model = model
        with tracer.start_as_current_span("call_llm"):
            wrapped(None, "e1", request, _chunk("a"), None)
            wrapped(None, "e1", request, _chunk("b"), None)

    assert tool.calls == 2  # once per span, not once per chunk

    spans = in_memory_span_exporter.get_finished_spans()
    assert [(s.attributes or {})[SpanAttributes.LLM_MODEL_NAME] for s in spans] == [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
    ]


def test_nothing_is_derived_for_a_non_recording_span() -> None:
    """A sampled-out span must not pay for attributes nobody will read."""
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider(sampler=ALWAYS_OFF)
    provider.add_span_processor(SimpleSpanProcessor(span_exporter=exporter))
    tool = _tool()
    request = _request(tool)
    tracer = _oi_tracer(provider)
    wrapped = _TraceCallLlm(tracer)(_noop_trace_call_llm)

    with tracer.start_as_current_span("call_llm"):
        for i in range(5):
            wrapped(None, "e1", request, _chunk(f"c{i}"), None)

    assert tool.calls == 0
    assert exporter.get_finished_spans() == ()


@pytest.mark.parametrize("chunks", [1, 5, 20])
def test_declaration_cost_does_not_scale_with_chunk_count(
    chunks: int,
    tracer_provider: trace_api.TracerProvider,
) -> None:
    """The whole point of the change: cost is O(1) in the number of chunks."""
    tool = _tool()
    request = _request(tool)
    tracer = _oi_tracer(tracer_provider)
    wrapped = _TraceCallLlm(tracer)(_noop_trace_call_llm)

    with tracer.start_as_current_span("call_llm"):
        for i in range(chunks):
            wrapped(None, "e1", request, _chunk(f"c{i}"), None)

    assert tool.calls == 1


def test_guard_survives_the_span_attribute_limit(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """A long request fills the span's bounded attribute dict and evicts the earliest keys,
    `INPUT_VALUE` among them. The guard must not depend on those surviving, or it would
    switch itself off for exactly the requests that are most expensive to re-derive."""
    tool = _tool()
    request = _request(tool, history=200)
    tracer = _oi_tracer(tracer_provider)
    wrapped = _TraceCallLlm(tracer)(_noop_trace_call_llm)

    with tracer.start_as_current_span("call_llm"):
        for i in range(5):
            wrapped(None, "e1", request, _chunk(f"c{i}"), None)

    span = in_memory_span_exporter.get_finished_spans()[0]
    attributes = dict(span.attributes or {})
    # Precondition: the limit really was hit and the obvious marker really is gone.
    assert len(attributes) == SpanLimits().max_attributes
    assert SpanAttributes.INPUT_VALUE not in attributes

    assert tool.calls == 1


def test_a_second_request_on_the_same_span_is_still_recorded(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """The guard is keyed on the request, not just the span. ADK binds one request per
    `call_llm` span today, but a span that ever serves a second, different request must
    still get that request's attributes rather than keeping the first one's."""
    tool = _tool()
    first = _request(tool)
    second = _request(tool)
    second.model = "gemini-2.5-flash"
    tracer = _oi_tracer(tracer_provider)
    wrapped = _TraceCallLlm(tracer)(_noop_trace_call_llm)

    with tracer.start_as_current_span("call_llm"):
        wrapped(None, "e1", first, _chunk("a"), None)
        wrapped(None, "e1", first, _chunk("b"), None)
        wrapped(None, "e2", second, _chunk("c"), None)
        wrapped(None, "e2", second, _chunk("d"), None)

    assert tool.calls == 2  # once per request, not once per chunk and not once per span

    span = in_memory_span_exporter.get_finished_spans()[0]
    assert (span.attributes or {})[SpanAttributes.LLM_MODEL_NAME] == "gemini-2.5-flash"


def test_an_escaping_exception_is_contained_and_the_pass_is_retried(
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """The span is marked only once the request-side pass has run without an *escaping*
    exception. Such an exception must not reach the caller -- instrumentation cannot be
    allowed to fail an otherwise successful ADK call -- and the next chunk must redo the
    pass rather than skip it for the rest of the span.

    The fault is injected by replacing an extractor with an undecorated function that
    raises, standing in for the unguarded code in `_set_request_attributes` (the
    `span.set_attribute` calls, which are not wrapped in `@stop_on_exception`). For errors
    raised *inside* the decorated extractors, see the single-pass test below."""
    tool = _tool()
    request = _request(tool)
    tracer = _oi_tracer(tracer_provider)
    wrapped = _TraceCallLlm(tracer)(_noop_trace_call_llm)

    real = _wrappers._get_attributes_from_base_tool
    fail = [True]

    def _boom(*args: Any, **kwargs: Any) -> Any:
        if fail[0]:
            fail[0] = False
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(_wrappers, "_get_attributes_from_base_tool", _boom)

    with tracer.start_as_current_span("call_llm"):
        wrapped(None, "e1", request, _chunk("a"), None)  # must not raise
        wrapped(None, "e1", request, _chunk("b"), None)
        wrapped(None, "e1", request, _chunk("c"), None)

    assert tool.calls == 1  # the retry succeeded, and only the retry did the work

    span = in_memory_span_exporter.get_finished_spans()[0]
    assert f"{SpanAttributes.LLM_TOOLS}.0.tool.json_schema" in (span.attributes or {})


def test_an_error_swallowed_by_an_extractor_is_retried_on_the_next_chunk(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """The attribute extractors are `@stop_on_exception`: they log and swallow, so a failed
    extraction never reaches the wrapper as an exception. The request must not be marked
    written in that case, or the attribute would stay missing for the whole span. Uses the
    real decorated path -- a tool whose declaration fails once and succeeds afterwards."""

    class _FlakyTool(FunctionTool):
        calls = 0

        def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
            type(self).calls += 1
            if type(self).calls == 1:
                raise RuntimeError("declaration unavailable")
            return super()._get_declaration()

    def search(query: str) -> dict[str, Any]:
        """A search tool.

        Args:
            query: The search query.
        """
        return {}

    _FlakyTool.calls = 0
    tool = _FlakyTool(func=search)
    request = _request(tool)
    tracer = _oi_tracer(tracer_provider)
    wrapped = _TraceCallLlm(tracer)(_noop_trace_call_llm)

    with tracer.start_as_current_span("call_llm"):
        wrapped(None, "e1", request, _chunk("a"), None)  # swallowed, so not marked written
        wrapped(None, "e1", request, _chunk("b"), None)  # retried, and succeeds
        wrapped(None, "e1", request, _chunk("c"), None)  # marked written, no further work

    assert tool.calls == 2  # the failure and the retry, not once per chunk

    attributes = dict(in_memory_span_exporter.get_finished_spans()[0].attributes or {})
    assert f"{SpanAttributes.LLM_TOOLS}.0.tool.json_schema" in attributes


def test_request_attributes_are_not_rederived_after_request_mutation(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Request-side attributes are derived only once, even if the request is later mutated."""
    tool = _tool()
    request = _request(tool)
    tracer = _oi_tracer(tracer_provider)
    wrapped = _TraceCallLlm(tracer)(_noop_trace_call_llm)

    with tracer.start_as_current_span("call_llm"):
        wrapped(None, "e1", request, _chunk("a"), None)

        request.model = "gemini-2.5-flash"
        wrapped(None, "e1", request, _chunk("b"), None)

    assert tool.calls == 1

    span = in_memory_span_exporter.get_finished_spans()[0]
    assert (span.attributes or {})[SpanAttributes.LLM_MODEL_NAME] == "gemini-2.0-flash"


class _FakeSpan:
    """Only `get_span_context().span_id` is used by the bookkeeping."""

    def __init__(self, span_id: int) -> None:
        self._span_id = span_id

    def get_span_context(self) -> Any:
        return SimpleNamespace(span_id=self._span_id)


def test_the_record_survives_concurrent_inserts_while_full(
    tracer_provider: trace_api.TracerProvider,
) -> None:
    """One wrapper is installed per process, so its record is shared by every thread
    running an ADK stream. Fill it to its bound, so every further insert also evicts, then
    insert new spans from several threads at once: instrumentation may never raise into an
    otherwise successful ADK call, and the bound must hold."""
    wrapper = _TraceCallLlm(_oi_tracer(tracer_provider))
    limit = _TraceCallLlm._MAX_REMEMBERED_SPANS
    workers = 8
    requests = [_request(_tool()) for _ in range(workers)]  # kept alive: the record is weak

    for span_id in range(limit):
        wrapper._remember_request_attributes(_FakeSpan(span_id), requests[0])

    errors: list[BaseException] = []
    barrier = threading.Barrier(workers)

    def hammer(worker: int) -> None:
        request = requests[worker]
        barrier.wait()
        try:
            for i in range(500):
                span = _FakeSpan(limit + worker * 500 + i)
                wrapper._remember_request_attributes(span, request)
                wrapper._request_attributes_written(span, request)
        except BaseException as exc:  # noqa: BLE001 - the point of the test
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(w,)) for w in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(wrapper._request_written_for_span) == limit


def test_the_lock_is_not_held_while_attributes_are_derived(
    tracer_provider: trace_api.TracerProvider,
) -> None:
    """The lock covers the bookkeeping only. Holding it across attribute derivation would
    serialize every concurrent ADK stream in the process, costing far more than the
    per-chunk work this wrapper exists to avoid."""
    tracer = _oi_tracer(tracer_provider)
    instance = _TraceCallLlm(tracer)
    wrapped = instance(_noop_trace_call_llm)
    held: list[bool] = []

    class _WatchingTool(_CountingTool):
        def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
            held.append(instance._lock.locked())
            return super()._get_declaration()

    def search(query: str) -> dict[str, Any]:
        """A search tool.

        Args:
            query: The search query.
        """
        return {}

    _WatchingTool.calls = 0
    with tracer.start_as_current_span("call_llm"):
        wrapped(None, "e1", _request(_WatchingTool(func=search)), _chunk("a"), None)

    assert held == [False]


def test_a_collected_request_never_matches_a_later_one(
    tracer_provider: trace_api.TracerProvider,
) -> None:
    """The record holds a weak reference, so an entry whose request has been collected can
    never be mistaken for a new request that happens to reuse its memory address."""
    wrapper = _TraceCallLlm(_oi_tracer(tracer_provider))
    span = _FakeSpan(1)

    request = _request(_tool())
    wrapper._remember_request_attributes(span, request)
    assert wrapper._request_attributes_written(span, request)

    del request
    gc.collect()

    successor = _request(_tool())
    assert not wrapper._request_attributes_written(span, successor)
