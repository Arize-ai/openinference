from typing import Any, Generator

import pytest
from openinference.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def setup_crewai_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    CrewAIInstrumentor().uninstrument()


def test_crewai_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_crewai_instrumentation: Any,
) -> None:
    # TODO(harrison) Figure out how to unit test this beast
    assert True
