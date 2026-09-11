from typing import Callable, Dict, Optional, Sequence, Union

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
from opentelemetry.trace import Link, SpanKind, TraceState
from opentelemetry.util.types import Attributes, AttributeValue

from openinference.instrumentation import REDACTED_VALUE, OITracer, TraceConfig, suppress_tracing
from openinference.semconv.trace import SpanAttributes


class _CapturingSampler(Sampler):
    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        self.attributes: Dict[str, AttributeValue] = {}

    def should_sample(
        self,
        parent_context: Optional[Context],
        trace_id: int,
        name: str,
        kind: Optional[SpanKind] = None,
        attributes: Attributes = None,
        links: Optional[Sequence[Link]] = None,
        trace_state: Optional[TraceState] = None,
    ) -> SamplingResult:
        self.attributes = dict(attributes or {})
        return SamplingResult(self.decision)

    def get_description(self) -> str:
        return "capturing sampler"


class _FinishedSpanCollector(SpanProcessor):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        self.spans.append(span)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _tracer(
    decision: Decision = Decision.RECORD_AND_SAMPLE,
    *,
    config: Optional[TraceConfig] = None,
) -> tuple[OITracer, _CapturingSampler, InMemorySpanExporter]:
    sampler = _CapturingSampler(decision)
    provider = TracerProvider(sampler=sampler)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OITracer(provider.get_tracer(__name__), config or TraceConfig()), sampler, exporter


@pytest.mark.parametrize("decision", [Decision.RECORD_ONLY, Decision.RECORD_AND_SAMPLE])
def test_deferred_input_value_is_finalized_for_recording_spans(decision: Decision) -> None:
    sampler = _CapturingSampler(decision)
    collector = _FinishedSpanCollector()
    provider = TracerProvider(sampler=sampler)
    provider.add_span_processor(collector)
    tracer = OITracer(provider.get_tracer(__name__), TraceConfig())
    callback_count = 0

    def input_value() -> str:
        nonlocal callback_count
        callback_count += 1
        return '{"messages":[{"content":"hello"}]}'

    span = tracer.start_span(
        "llm",
        attributes={SpanAttributes.INPUT_VALUE: "unredacted sampler value"},
        input_value=input_value,
    )
    span.end()

    assert sampler.attributes[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert callback_count == 1
    (finished_span,) = collector.spans
    assert finished_span.attributes is not None
    assert (
        finished_span.attributes[SpanAttributes.INPUT_VALUE] == '{"messages":[{"content":"hello"}]}'
    )


def test_deferred_input_value_is_not_finalized_for_dropped_span() -> None:
    tracer, sampler, exporter = _tracer(Decision.DROP)
    callback_count = 0

    def input_value() -> str:
        nonlocal callback_count
        callback_count += 1
        return "visible"

    tracer.start_span("llm", input_value=input_value).end()

    assert sampler.attributes[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert callback_count == 0
    assert exporter.get_finished_spans() == ()


def test_deferred_input_value_is_not_finalized_when_inputs_are_hidden() -> None:
    tracer, sampler, exporter = _tracer(config=TraceConfig(hide_inputs=True))

    def input_value() -> str:
        raise AssertionError("hidden input callback ran")

    tracer.start_span("llm", input_value=input_value).end()

    assert sampler.attributes[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    (exported_span,) = exporter.get_finished_spans()
    assert exported_span.attributes is not None
    assert exported_span.attributes[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE


def test_deferred_input_value_is_not_finalized_for_suppressed_span() -> None:
    tracer, _, exporter = _tracer()

    def input_value() -> str:
        raise AssertionError("suppressed input callback ran")

    with suppress_tracing():
        tracer.start_span("llm", input_value=input_value).end()

    assert exporter.get_finished_spans() == ()


def test_deferred_input_value_failure_leaves_redaction() -> None:
    tracer, _, exporter = _tracer()

    def input_value() -> str:
        raise RuntimeError("serialization failed")

    tracer.start_span("llm", input_value=input_value).end()

    (exported_span,) = exporter.get_finished_spans()
    assert exported_span.attributes is not None
    assert exported_span.attributes[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE


def test_start_as_current_span_forwards_deferred_input_value() -> None:
    tracer, sampler, exporter = _tracer()

    with tracer.start_as_current_span("llm", input_value=lambda: "final"):
        pass

    assert sampler.attributes[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    (exported_span,) = exporter.get_finished_spans()
    assert exported_span.attributes is not None
    assert exported_span.attributes[SpanAttributes.INPUT_VALUE] == "final"


def test_deferred_input_value_is_not_finalized_for_invalid_span() -> None:
    tracer = OITracer(trace_api.NoOpTracerProvider().get_tracer(__name__), TraceConfig())

    def input_value() -> str:
        raise AssertionError("invalid span input callback ran")

    tracer.start_span("llm", input_value=input_value).end()


def test_deferred_input_value_is_not_finalized_after_on_start_ends_span() -> None:
    class _EndingSpanProcessor(SpanProcessor):
        def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
            span.end()

        def on_end(self, span: ReadableSpan) -> None:
            pass

        def shutdown(self) -> None:
            pass

        def force_flush(self, timeout_millis: int = 30_000) -> bool:
            return True

    provider = TracerProvider()
    provider.add_span_processor(_EndingSpanProcessor())
    tracer = OITracer(provider.get_tracer(__name__), TraceConfig())

    def input_value() -> str:
        raise AssertionError("ended span input callback ran")

    tracer.start_span("llm", input_value=input_value)


def test_deferred_input_value_never_reaches_custom_mask() -> None:
    received_values: list[AttributeValue] = []

    class _CustomTraceConfig(TraceConfig):
        def mask(
            self,
            key: str,
            value: Union[AttributeValue, Callable[[], AttributeValue]],
            *,
            externalize: bool = True,
        ) -> Optional[AttributeValue]:
            assert not callable(value)
            received_values.append(value)
            return super().mask(key, value, externalize=externalize)

    tracer, sampler, exporter = _tracer(config=_CustomTraceConfig())
    tracer.start_span("llm", input_value=lambda: "final").end()

    assert sampler.attributes[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert received_values == [REDACTED_VALUE, REDACTED_VALUE, "final"]
    (exported_span,) = exporter.get_finished_spans()
    assert exported_span.attributes is not None
    assert exported_span.attributes[SpanAttributes.INPUT_VALUE] == "final"
