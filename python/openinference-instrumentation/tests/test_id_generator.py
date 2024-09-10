from random import seed

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import OITracer, TraceConfig


def test_id_generator_is_unaffected_by_seed() -> None:
    in_memory_span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    tracer = OITracer(tracer_provider.get_tracer(__name__), TraceConfig())
    n = 10
    for _ in range(n):
        seed(42)
        tracer.start_span("test").end()
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == n
    assert len(set(span.context.trace_id for span in spans)) == n
    assert len(set(span.context.span_id for span in spans)) == n
