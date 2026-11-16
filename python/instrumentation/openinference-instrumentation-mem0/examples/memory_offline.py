"""Traces mem0 memory operations without any API key.

mem0's LLM and embedder are replaced by small in-process stubs and the vector store is an
in-memory Qdrant, so this runs fully offline. Spans are exported to Phoenix at
http://localhost:6006 under the project "mem0-offline".
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

os.environ.setdefault("MEM0_TELEMETRY", "False")

from mem0 import Memory  # noqa: E402
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: E402
from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from openinference.instrumentation import using_attributes  # noqa: E402
from openinference.instrumentation.mem0 import Mem0Instrumentor  # noqa: E402
from openinference.semconv.resource import ResourceAttributes  # noqa: E402

endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006") + "/v1/traces"
tracer_provider = TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: "mem0-offline"})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
Mem0Instrumentor().instrument(tracer_provider=tracer_provider)

DIMS = 64


class StubEmbedder:
    """Bag-of-words hashing embedder: deterministic and good enough for keyword overlap."""

    def embed(self, text: Any, memory_action: Optional[str] = None) -> List[float]:
        vector = [0.0] * DIMS
        for token in str(text).lower().replace("?", " ").replace(".", " ").split():
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIMS] += 1.0
        return vector

    def embed_batch(
        self, texts: List[str], memory_action: Optional[str] = None
    ) -> List[List[float]]:
        return [self.embed(text, memory_action) for text in texts]


class StubLLM:
    """Plays the fact-extraction LLM: keeps each first-person user sentence as a fact."""

    def generate_response(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        prompt = messages[-1]["content"]
        new_messages = prompt.split("## New Messages", 1)[-1].split("##", 1)[0]
        facts = []
        for line in new_messages.splitlines():
            if line.startswith("user:"):
                for sentence in line[len("user:") :].split("."):
                    if sentence.strip().startswith("I "):
                        facts.append({"text": sentence.strip()})
        return json.dumps({"memory": facts})


memory = Memory.from_config(
    {
        "vector_store": {
            "provider": "qdrant",
            "config": {"client": QdrantClient(":memory:"), "embedding_model_dims": DIMS},
        },
        # Only constructed, never called: replaced by the stubs below.
        "llm": {"provider": "openai", "config": {"api_key": "unused"}},
        "embedder": {
            "provider": "openai",
            "config": {"api_key": "unused", "embedding_dims": DIMS},
        },
        "history_db_path": ":memory:",
    }
)
memory.llm = StubLLM()
memory.embedding_model = StubEmbedder()

if __name__ == "__main__":
    with using_attributes(session_id="mem0-demo-session"):
        added = memory.add(
            [
                {"role": "user", "content": "I love playing tennis. I am vegetarian."},
                {"role": "assistant", "content": "Noted! Tennis and vegetarian food."},
            ],
            user_id="alice",
            agent_id="concierge",
        )
        print("add:", added)
        memory.add("Prefers window seats", user_id="alice", infer=False)

        hits = memory.search("does alice like tennis", filters={"user_id": "alice"}, top_k=3)
        print("search:", hits)
        print("get_all:", memory.get_all(filters={"user_id": "alice"}))

        memory_id = added["results"][0]["id"]
        print("get:", memory.get(memory_id))
        memory.update(memory_id, text="User plays tennis every Sunday")
        print("history:", memory.history(memory_id))
        memory.delete(memory_id)
        memory.delete_all(user_id="alice")

        try:
            memory.get_all(filters={})  # mem0 requires a user_id/agent_id/run_id scope
        except ValueError as error:
            print("expected error:", error)
