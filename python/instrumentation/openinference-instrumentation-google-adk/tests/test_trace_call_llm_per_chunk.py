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


def test_a_failed_first_pass_is_retried_rather_than_suppressed(
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """The span is marked only once the request-side block has completed. If a pass raises
    part way through, the next chunk must redo it rather than skip it for the rest of the
    span. (The attribute extractors are `@stop_on_exception`, so this forces the case.)"""
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
        with pytest.raises(RuntimeError):
            wrapped(None, "e1", request, _chunk("a"), None)
        wrapped(None, "e1", request, _chunk("b"), None)
        wrapped(None, "e1", request, _chunk("c"), None)

    assert tool.calls == 1  # the retry succeeded, and only the retry did the work

    span = in_memory_span_exporter.get_finished_spans()[0]
    assert f"{SpanAttributes.LLM_TOOLS}.0.tool.json_schema" in (span.attributes or {})
