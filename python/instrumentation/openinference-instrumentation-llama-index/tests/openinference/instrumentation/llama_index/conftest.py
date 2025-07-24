from typing import Iterator
from unittest.mock import Mock, patch

import google.auth.credentials
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> TracerProvider:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture(autouse=True)
def uninstrument() -> Iterator[None]:
    yield
    LlamaIndexInstrumentor().uninstrument()


@pytest.fixture(autouse=True)
def openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-")


@pytest.fixture(autouse=True)
def anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-")


@pytest.fixture(autouse=True)
def mock_google_auth() -> Iterator[Mock]:
    """Mock Google authentication to prevent network calls during testing."""
    with patch("google.auth.default") as mock_auth:
        mock_credentials = Mock(spec=google.auth.credentials.Credentials)
        mock_credentials.token = "fake_token"
        mock_credentials.valid = True
        mock_credentials.expired = False
        mock_auth.return_value = (mock_credentials, "fake-project")
        yield mock_auth
