import warnings
from typing import Any, Callable

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider as OTelTracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from openinference.instrumentation import (
    OITracer,
    TracerProvider,
    agent_span,
    chain_span,
    evaluator_span,
    get_span_kind_attributes,
    guardrail_span,
    llm_span,
    reranker_span,
    retriever_span,
    tool_span,
)
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
INPUT_VALUE = SpanAttributes.INPUT_VALUE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE


@pytest.mark.parametrize(
    "decorator,kind",
    [
        (agent_span, OpenInferenceSpanKindValues.AGENT),
        (chain_span, OpenInferenceSpanKindValues.CHAIN),
        (retriever_span, OpenInferenceSpanKindValues.RETRIEVER),
        (reranker_span, OpenInferenceSpanKindValues.RERANKER),
        (guardrail_span, OpenInferenceSpanKindValues.GUARDRAIL),
        (evaluator_span, OpenInferenceSpanKindValues.EVALUATOR),
        (tool_span, OpenInferenceSpanKindValues.TOOL),
        (llm_span, OpenInferenceSpanKindValues.LLM),
    ],
)
def test_span_decorators_with_explicit_tracer(
    decorator: Callable[..., Any],
    kind: OpenInferenceSpanKindValues,
    tracer: OITracer,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    def run(text: str) -> str:
        return text.upper()

    run = decorator(tracer=tracer, name="custom-name")(run)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert run("hi") == "HI"

    (span,) = in_memory_span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert span.name == "custom-name"
    assert span.status.status_code is StatusCode.OK
    assert attributes[OPENINFERENCE_SPAN_KIND] == kind.value
    assert "hi" in str(attributes[INPUT_VALUE])
    assert attributes[OUTPUT_VALUE] == "HI"


def test_span_decorator_defaults_to_global_tracer_provider() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    @chain_span
    def run(text: str) -> str:
        return text

    original = trace_api.get_tracer_provider()
    trace_api._TRACER_PROVIDER = provider  # bypass the set-once guard for test isolation
    try:
        assert run("hi") == "hi"
    finally:
        trace_api._TRACER_PROVIDER = original

    (span,) = exporter.get_finished_spans()
    assert span.name == "run"
    assert (span.attributes or {})[OPENINFERENCE_SPAN_KIND] == "CHAIN"


def test_span_decorator_wraps_plain_otel_tracer() -> None:
    exporter = InMemorySpanExporter()
    provider = OTelTracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    plain_tracer = provider.get_tracer(__name__)
    assert not isinstance(plain_tracer, OITracer)

    @tool_span(tracer=plain_tracer)
    def add(a: int, b: int) -> int:
        """Adds two numbers."""
        return a + b

    assert add(1, 2) == 3
    (span,) = exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes[OPENINFERENCE_SPAN_KIND] == "TOOL"
    assert attributes[SpanAttributes.TOOL_DESCRIPTION] == "Adds two numbers."
    assert attributes[OUTPUT_VALUE] == "3"


@pytest.mark.parametrize(
    "method", ["agent", "chain", "retriever", "reranker", "guardrail", "evaluator", "tool", "llm"]
)
def test_tracer_decorator_methods_warn_and_still_work(
    method: str,
    tracer: OITracer,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    def run(text: str) -> str:
        return text

    with pytest.warns(DeprecationWarning, match=f"`OITracer.{method}` is deprecated"):
        run = getattr(tracer, method)(run)

    with pytest.warns(DeprecationWarning, match=f"@{method}_span\\(tracer=tracer\\)"):
        getattr(tracer, method)(name="n")

    assert run("hi") == "hi"
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert (span.attributes or {})[OPENINFERENCE_SPAN_KIND] == method.upper()


def test_openinference_span_kind_argument_warns_and_still_sets_kind(
    tracer: OITracer,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with pytest.warns(DeprecationWarning, match="`openinference_span_kind` is deprecated"):
        with tracer.start_as_current_span("a", openinference_span_kind="chain"):
            pass
    with pytest.warns(DeprecationWarning, match="get_span_kind_attributes"):
        tracer.start_span(
            "b",
            openinference_span_kind="llm",
            attributes={OPENINFERENCE_SPAN_KIND: "TOOL", "k": "v"},
        ).end()

    a, b = in_memory_span_exporter.get_finished_spans()
    assert (a.attributes or {})[OPENINFERENCE_SPAN_KIND] == "CHAIN"
    assert dict(b.attributes or {}) == {OPENINFERENCE_SPAN_KIND: "LLM", "k": "v"}


def test_span_kind_via_attributes_does_not_warn(
    tracer: OITracer,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with tracer.start_as_current_span("a", attributes=get_span_kind_attributes("agent")):
            pass
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert (span.attributes or {})[OPENINFERENCE_SPAN_KIND] == "AGENT"
