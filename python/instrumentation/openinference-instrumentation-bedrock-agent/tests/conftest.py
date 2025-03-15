from typing import Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.bedrock_agent import BedrockAgentInstrumentor


@pytest.fixture(scope="function", autouse=True)
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> TracerProvider:
    tracer_provider = TracerProvider()
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


@pytest.fixture(scope="function", autouse=True)
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(scope="function", autouse=True)
def instrument(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[None]:
    BedrockAgentInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    BedrockAgentInstrumentor().uninstrument()
