from typing import Generator

import crewai
import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from packaging import version

from openinference.instrumentation.crewai import CrewAIInstrumentor


@pytest.fixture(scope="session")
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(scope="session")
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> trace_api.TracerProvider:
    tracer_provider = trace_sdk.TracerProvider()
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    in_memory_span_exporter.clear()
    yield
    CrewAIInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


@pytest.fixture(scope="function")
def vcr_cassette_name(request: pytest.FixtureRequest) -> str:
    crewai_version = version.parse(crewai.__version__)
    function_name = request.function.__name__
    if crewai_version >= version.parse("1.9.0"):
        suffix = "_latest"
    else:
        suffix = "_legacy"
    return f"{function_name}{suffix}"
