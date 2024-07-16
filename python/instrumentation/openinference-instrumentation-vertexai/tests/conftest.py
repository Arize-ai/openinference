import logging
from typing import Generator, cast

import pytest
from openinference.instrumentation.vertexai import VertexAIInstrumentor
from opentelemetry.sdk.trace import Tracer, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

for name, logger in logging.root.manager.loggerDict.items():
    if name.startswith("openinference.") and isinstance(logger, logging.Logger):
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler())


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    (tracer_provider := TracerProvider()).add_span_processor(
        SimpleSpanProcessor(in_memory_span_exporter)
    )
    return tracer_provider


@pytest.fixture
def tracer(tracer_provider: TracerProvider) -> Tracer:
    return cast(Tracer, tracer_provider.get_tracer(__name__))


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    VertexAIInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    VertexAIInstrumentor().uninstrument()
    in_memory_span_exporter.clear()
