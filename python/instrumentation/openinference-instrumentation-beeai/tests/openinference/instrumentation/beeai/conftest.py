from typing import Any, Generator

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.beeai import BeeAIInstrumentor


@pytest.fixture(scope="session")
def vcr_config() -> dict[str, Any]:
    return {
        "record_mode": "once",
        "filter_headers": [
            "authorization",
            "api-key",
            "x-api-key",
        ],
        # Match requests on these attributes
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
        # Decode compressed responses
        "decode_compressed_response": True,
        # Allow recording of requests
        "allow_playback_repeats": True,
    }


@pytest.fixture
def openai_api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    api_key = "sk-fake-key"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    return api_key


@pytest.fixture
def serperdev_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERPERDEV_API_KEY", "sk-fake-key")


@pytest.fixture
def cohere_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "sk-fake-key")


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
async def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> trace_api.TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    instrumentor = BeeAIInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    yield
    instrumentor.uninstrument()
    in_memory_span_exporter.clear()
