from typing import Generator

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.pydantic_ai.span_processor import OpenInferenceSpanProcessor


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
    tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
    return tracer_provider


@pytest.fixture(autouse=True)
def clear_spans(
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    in_memory_span_exporter.clear()
    yield
    in_memory_span_exporter.clear()
