"""Pytest configuration and fixtures for Strands instrumentation tests."""

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def tracer_provider():
    """Create a tracer provider with in-memory span exporter for testing."""
    tracer_provider = trace_sdk.TracerProvider()
    return tracer_provider


@pytest.fixture
def in_memory_span_exporter():
    """Create an in-memory span exporter for testing."""
    return InMemorySpanExporter()


@pytest.fixture
def instrumented_tracer_provider(tracer_provider, in_memory_span_exporter):
    """Create an instrumented tracer provider with in-memory exporter."""
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    trace_api.set_tracer_provider(tracer_provider)
    return tracer_provider

