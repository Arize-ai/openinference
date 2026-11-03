"""Pytest configuration and fixtures for Strands instrumentation tests."""

import os
from typing import Any

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def tracer_provider() -> trace_sdk.TracerProvider:
    """Create a tracer provider with in-memory span exporter for testing."""
    tracer_provider = trace_sdk.TracerProvider()
    return tracer_provider


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    """Create an in-memory span exporter for testing."""
    return InMemorySpanExporter()


@pytest.fixture
def instrumented_tracer_provider(
    tracer_provider: trace_sdk.TracerProvider, in_memory_span_exporter: InMemorySpanExporter
) -> trace_sdk.TracerProvider:
    """Create an instrumented tracer provider with in-memory exporter."""
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    trace_api.set_tracer_provider(tracer_provider)
    return tracer_provider


def _strip_request_headers(request: Any) -> Any:
    request.headers.clear()
    return request


def _strip_response_headers(response: Any) -> Any:
    return {**response, "headers": {}}


@pytest.fixture(scope="session")
def vcr_config() -> dict[str, Any]:
    return {
        "before_record_request": _strip_request_headers,
        "before_record_response": _strip_response_headers,
        "decode_compressed_response": True,
        "record_mode": "once",
    }


@pytest.fixture(autouse=True)
def openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recorded cassettes replay without a key; a real one is only needed to record."""
    if not os.environ.get("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
