import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import OITracer, TraceConfig, TracerProvider


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def trace_config() -> TraceConfig:
    return TraceConfig()


@pytest.fixture
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
    trace_config: TraceConfig,
) -> TracerProvider:
    tracer_provider = TracerProvider(config=trace_config)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture
def tracer(tracer_provider: TracerProvider) -> OITracer:
    return tracer_provider.get_tracer(__name__)
