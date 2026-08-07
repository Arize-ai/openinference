import os
from typing import Any, Dict, Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.together import TogetherInstrumentor


def _strip_request_headers(request: Any) -> Any:
    request.headers.clear()
    return request


def _strip_response_headers(response: Dict[str, Any]) -> Dict[str, Any]:
    return {**response, "headers": {}}


@pytest.fixture(scope="session")
def vcr_config() -> Dict[str, Any]:
    return {
        "before_record_request": _strip_request_headers,
        "before_record_response": _strip_response_headers,
        "decode_compressed_response": True,
        "record_mode": "once",
    }


@pytest.fixture(autouse=True)
def together_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use the real key when recording cassettes; fall back to a placeholder
    # so cassette replay works without credentials (e.g. in CI).
    monkeypatch.setenv("TOGETHER_API_KEY", os.environ.get("TOGETHER_API_KEY", "fake-api-key"))


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(tracer_provider: TracerProvider) -> Iterator[None]:
    TogetherInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    TogetherInstrumentor().uninstrument()
