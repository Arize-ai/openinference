from typing import Any, Generator

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from __init__ import GroqInstrumentor  #CHANGE
from groq.resources.chat.completions import AsyncCompletions, Completions


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
def setup_groq_instrumentation(
        tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    GroqInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    GroqInstrumentor().uninstrument()


def test_groq_instrumentation(
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_groq_instrumentation: Any,
) -> None:
    spans = in_memory_span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "Completions"
    ]

def test_groq_async_instrumentation(
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_groq_instrumentation: Any,
) -> None:
    spans = in_memory_span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "AsyncCompletions"
    ]

def test_groq_uninstrumentation(
        tracer_provider: TracerProvider,
) -> None:
    # Instrumenting Groq
    GroqInstrumentor().instrument(tracer_provider=tracer_provider)

    # Ensure methods are wrapped
    assert hasattr(Completions.create, "__wrapped__")
    assert hasattr(AsyncCompletions.create, "__wrapped__")

    # Uninstrumenting Groq
    GroqInstrumentor().uninstrument()

    # Ensure methods are not wrapped
    assert not hasattr(Completions.create, "__wrapped__")
    assert not hasattr(AsyncCompletions.create, "__wrapped__")
