import json
from typing import Any, Dict, Iterator, List

import httpx2
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from typesafe_sdk import TypeSafeClient

from openinference.instrumentation.typesafe import TypeSafeAIInstrumentor

# A canned System One response matching the wire format documented at
# https://docs.typesafe.ai/api. Answers cover all three primitives.
SYSTEM_ONE_RESPONSE: Dict[str, Any] = {
    "model": "jev-1.13.0",
    "answers": {
        "billing": {"type": "noul", "noul": 0.98},
        "tone": {
            "type": "choice",
            "choice": "angry",
            "confidence": 0.96,
            "probabilities": {"calm": 0.02, "angry": 0.98},
        },
        "urgency": {
            "type": "score",
            "score": 1.96,
            "confidence": 0.94,
            "legend": {"0": "low", "1": "medium", "2": "high"},
            "probabilities": {"0": 0.0, "1": 0.04, "2": 0.96},
        },
    },
    "usage": {"input_tokens": 344, "output_tokens": 65},
}


class RecordingTransport:
    """Serves canned System One JSON and records every request body it is handed.

    The SDK speaks ``httpx2``, which vcrpy does not patch, so tests inject the
    ``httpx2.MockTransport`` this exposes instead of recording cassettes.
    """

    def __init__(self, status_code: int = 200, body: Any = None) -> None:
        self.requests: List[Dict[str, Any]] = []
        self._status_code = status_code
        self._body = SYSTEM_ONE_RESPONSE if body is None else body

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        return httpx2.Response(
            self._status_code,
            json=self._body,
            headers={"x-typesafe-request-id": "req_test"},
        )

    @property
    def mock(self) -> httpx2.MockTransport:
        """Returns a transport that both the sync and async clients accept."""
        return httpx2.MockTransport(self._handle)


@pytest.fixture(autouse=True)
def typesafe_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-api-key")
    monkeypatch.delenv("TYPESAFE_DEFAULT_MODEL", raising=False)


@pytest.fixture()
def transport() -> RecordingTransport:
    return RecordingTransport()


@pytest.fixture()
def client(transport: RecordingTransport) -> TypeSafeClient:
    return TypeSafeClient(transport=transport.mock)


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
    TypeSafeAIInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    TypeSafeAIInstrumentor().uninstrument()
