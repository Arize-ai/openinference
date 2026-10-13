"""Span-ending behaviour of OpenInferenceSpan."""

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ParentBased

from openinference.instrumentation import (
    TraceConfig,
    TracerProvider,
    suppress_tracing,
)


def test_ending_a_suppressed_span_does_not_raise(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Instrumentation suppression yields a NonRecordingSpan, which has no
    attributes for the GenAI bridge to read."""
    tracer_provider = TracerProvider(config=TraceConfig(enable_genai_semconv=True))
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    tracer = tracer_provider.get_tracer(__name__)

    with suppress_tracing(), tracer.start_as_current_span("suppressed-span"):
        pass

    assert not in_memory_span_exporter.get_finished_spans()


def test_ending_a_dropped_span_does_not_raise(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """A span the sampler drops is a NonRecordingSpan too."""
    tracer_provider = TracerProvider(
        sampler=ParentBased(root=ALWAYS_OFF),
        config=TraceConfig(enable_genai_semconv=True),
    )
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    tracer = tracer_provider.get_tracer(__name__)

    with tracer.start_as_current_span("dropped-span"):
        pass

    assert not in_memory_span_exporter.get_finished_spans()
