"""Pytest configuration and fixtures for agent-framework integration tests."""

import os
from typing import Generator

import pytest
from agent_framework.observability import enable_instrumentation
from agent_framework.openai import OpenAIChatClient
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.agent_framework import AgentFrameworkToOpenInferenceProcessor


@pytest.fixture(scope="session")
def vcr_config():
    """pytest-recording configuration to sanitize recorded cassettes."""
    return {
        "filter_headers": ["authorization", "api-key", "x-api-key"],
        "decode_compressed_response": True,
        "record_mode": "once",  # Use existing cassettes, only record if missing
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
    }


@pytest.fixture(scope="session")
def in_memory_span_exporter() -> InMemorySpanExporter:
    """Create an in-memory span exporter for testing."""
    return InMemorySpanExporter()


@pytest.fixture(scope="session")
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> trace_api.TracerProvider:
    """Create a tracer provider with OpenInference processor and span exporter."""
    # Create our tracer provider FIRST
    tracer_provider = trace_sdk.TracerProvider()

    # Add OpenInference processor to transform spans
    tracer_provider.add_span_processor(AgentFrameworkToOpenInferenceProcessor())

    # Add exporter to capture spans
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)

    # Set the tracer provider globally BEFORE enabling instrumentation
    trace_api.set_tracer_provider(tracer_provider)

    # Now enable agent-framework instrumentation - it will use the existing global provider
    enable_instrumentation(enable_sensitive_data=True)

    return tracer_provider


@pytest.fixture(autouse=True)
def clear_spans(
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    """Clear spans before and after each test."""
    in_memory_span_exporter.clear()
    yield
    in_memory_span_exporter.clear()


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    """Get OpenAI API key from environment or use placeholder for VCR playback."""
    return os.getenv(
        "OPENAI_API_KEY",
        "sk-proj",
    )


@pytest.fixture
def openai_client(openai_api_key: str) -> OpenAIChatClient:
    """Create OpenAI chat client with default model."""
    return OpenAIChatClient(model_id="gpt-4o-mini", api_key=openai_api_key)


@pytest.fixture
def gpt4_client(openai_api_key: str) -> OpenAIChatClient:
    """Create OpenAI chat client with GPT-4o model."""
    return OpenAIChatClient(model_id="gpt-4o", api_key=openai_api_key)
