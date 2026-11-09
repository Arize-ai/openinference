import hashlib
import json
import os
from typing import Any, Dict, Iterator, List, Optional

# mem0 reads this at import time; keep its product telemetry off during tests.
os.environ.setdefault("MEM0_TELEMETRY", "False")

import httpx  # noqa: E402
import pytest  # noqa: E402
from mem0 import AsyncMemory, AsyncMemoryClient, Memory, MemoryClient  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from qdrant_client import QdrantClient  # noqa: E402

from openinference.instrumentation.mem0 import Mem0Instrumentor  # noqa: E402

EMBEDDING_DIMS = 16
EXTRACTED_FACT = "Likes to play tennis on weekends"


class FakeEmbedder:
    """Deterministic bag-of-words embedder so similarity search behaves sensibly offline."""

    def embed(self, text: Any, memory_action: Optional[str] = None) -> List[float]:
        vector = [0.0] * EMBEDDING_DIMS
        for token in str(text).lower().split():
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % EMBEDDING_DIMS] += 1.0
        return vector

    def embed_batch(
        self, texts: List[str], memory_action: Optional[str] = None
    ) -> List[List[float]]:
        return [self.embed(text, memory_action) for text in texts]


class FakeLLM:
    """Stands in for mem0's fact-extraction LLM call; returns one canned fact."""

    def __init__(self) -> None:
        self.calls = 0

    def generate_response(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        self.calls += 1
        return json.dumps({"memory": [{"text": EXTRACTED_FACT}]})


def _config() -> Dict[str, Any]:
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "client": QdrantClient(":memory:"),
                "embedding_model_dims": EMBEDDING_DIMS,
            },
        },
        # The OpenAI providers are only constructed (no request is made); they are
        # replaced by the fakes above right after the memory is created.
        "llm": {"provider": "openai", "config": {"api_key": "fake-api-key"}},
        "embedder": {
            "provider": "openai",
            "config": {"api_key": "fake-api-key", "embedding_dims": EMBEDDING_DIMS},
        },
        "history_db_path": ":memory:",
    }


@pytest.fixture()
def memory() -> Memory:
    memory = Memory.from_config(_config())
    memory.llm = FakeLLM()
    memory.embedding_model = FakeEmbedder()
    return memory


@pytest.fixture()
def async_memory() -> AsyncMemory:
    memory = AsyncMemory.from_config(_config())
    memory.llm = FakeLLM()
    memory.embedding_model = FakeEmbedder()
    return memory


PLATFORM_MEMORY: Dict[str, Any] = {
    "id": "mem-1",
    "memory": EXTRACTED_FACT,
    "user_id": "alice",
    "score": 0.9,
}


def _platform_response(request: httpx.Request) -> httpx.Response:
    """Serves canned Mem0 platform API responses for the endpoints the tests use."""
    path = request.url.path
    if path == "/v1/ping/":
        body: Any = {"org_id": "org", "project_id": "proj", "user_email": "test@example.com"}
    elif path == "/v3/memories/add/":
        body = {"results": [{"id": "mem-1", "memory": EXTRACTED_FACT, "event": "ADD"}]}
    elif path == "/v3/memories/search/":
        body = {"results": [PLATFORM_MEMORY]}
    elif path == "/v1/memories/missing/":
        return httpx.Response(404, json={"detail": "Memory not found"})
    else:
        body = {"message": "ok"}
    return httpx.Response(200, json=body)


@pytest.fixture()
def memory_client() -> MemoryClient:
    return MemoryClient(
        api_key="fake-api-key",
        client=httpx.Client(transport=httpx.MockTransport(_platform_response)),
    )


@pytest.fixture()
def async_memory_client(monkeypatch: pytest.MonkeyPatch) -> AsyncMemoryClient:
    # The async client validates its key with a blocking `requests` call at construction.
    def validate_api_key(self: AsyncMemoryClient) -> str:
        self.org_id, self.project_id = "org", "proj"
        return "test@example.com"

    monkeypatch.setattr(AsyncMemoryClient, "_validate_api_key", validate_api_key)
    return AsyncMemoryClient(
        api_key="fake-api-key",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_platform_response)),
    )


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
    Mem0Instrumentor().instrument(tracer_provider=tracer_provider)
    yield
    Mem0Instrumentor().uninstrument()
