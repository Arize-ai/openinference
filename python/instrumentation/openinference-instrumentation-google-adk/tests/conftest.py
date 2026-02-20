from typing import Any, Dict, Iterator, Protocol

import pytest
from _pytest.monkeypatch import MonkeyPatch
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.google_adk import GoogleADKInstrumentor


class _VcrRequest(Protocol):
    method: str


def method_case_insensitive(r1: _VcrRequest, r2: _VcrRequest) -> bool:
    return r1.method.lower() == r2.method.lower()


@pytest.fixture(scope="session")
def vcr_config() -> Dict[str, Any]:
    return {
        "record_mode": "once",
        "match_on": ["scheme", "host", "port", "path", "query", "method"],
        "custom_matchers": {
            "method": method_case_insensitive,
        },
    }


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> trace_api.TracerProvider:
    tracer_provider = trace_sdk.TracerProvider()
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


@pytest.fixture
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[None]:
    GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    GoogleADKInstrumentor().uninstrument()


@pytest.fixture(autouse=True)
def api_key(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "xyz")
