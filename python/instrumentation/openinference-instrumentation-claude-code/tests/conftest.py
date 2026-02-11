import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def in_memory_span_exporter():
    """Create an in-memory span exporter for testing."""
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(in_memory_span_exporter):
    """Create a tracer provider with in-memory exporter."""
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture(autouse=True)
def reset_global_tracer_provider():
    """Reset global tracer provider after each test."""
    yield
    trace_api._TRACER_PROVIDER = None


@pytest.fixture
def instrumentor():
    """Create and clean up instrumentor."""
    from openinference.instrumentation.claude_code import ClaudeCodeInstrumentor

    instrumentor = ClaudeCodeInstrumentor()
    yield instrumentor
    if hasattr(instrumentor, "_is_instrumented") and instrumentor._is_instrumented:
        instrumentor.uninstrument()
