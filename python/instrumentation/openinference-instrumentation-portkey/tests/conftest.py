import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def tracer_provider() -> TracerProvider:
    return TracerProvider()


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def setup_portkey_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    from openinference.instrumentation.portkey import PortkeyInstrumentor

    tracer_provider.add_span_processor(in_memory_span_exporter)
    PortkeyInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    PortkeyInstrumentor().uninstrument() 