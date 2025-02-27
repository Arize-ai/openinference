from random import seed

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

from openinference.instrumentation import OITracer, TraceConfig


@pytest.mark.parametrize(
    "override",
    [
        pytest.param(True, id="should-allow-user-override"),
        pytest.param(False, id="should-be-unaffected-by-seed"),
    ],
)
def test_id_generator(override: bool) -> None:
    in_memory_span_exporter = InMemorySpanExporter()
    tracer_provider = (
        TracerProvider(id_generator=type("Foo", (RandomIdGenerator,), {})())
        if override
        else TracerProvider()
    )
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    tracer = tracer_provider.get_tracer(__name__)
    oi_tracer = OITracer(tracer, TraceConfig())
    n = 10
    for tr in (tracer, oi_tracer):
        for _ in range(n):
            seed(42)
            with tr.start_as_current_span("parent"):
                tr.start_span("child").end()
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == n * 2 * 2
    assert len(set(span.context.trace_id for span in spans)) == 1 if override else (n + 1)
    assert len(set(span.context.span_id for span in spans)) == 2 if override else (n + 1) * 2
