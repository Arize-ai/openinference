from typing import Generator

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture
def setup_portkey_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    from openinference.instrumentation.portkey import PortkeyInstrumentor

    PortkeyInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    PortkeyInstrumentor().uninstrument()
