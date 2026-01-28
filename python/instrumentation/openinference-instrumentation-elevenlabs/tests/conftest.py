"""Pytest configuration and shared fixtures for ElevenLabs instrumentation tests."""

from typing import Generator

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.elevenlabs import ElevenLabsInstrumentor


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    """Create an in-memory span exporter for testing."""
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    """Create a TracerProvider configured with in-memory span export."""
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def tracer(tracer_provider: TracerProvider) -> OITracer:
    """Create an OITracer instance for testing."""
    config = TraceConfig()
    return OITracer(
        trace_api.get_tracer(__name__, "1.0.0", tracer_provider),
        config=config,
    )


@pytest.fixture()
def config() -> TraceConfig:
    """Create a TraceConfig instance for testing."""
    return TraceConfig()


@pytest.fixture()
def setup_elevenlabs_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    """Set up and tear down ElevenLabs instrumentation for tests."""
    instrumentor = ElevenLabsInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    yield
    instrumentor.uninstrument()
