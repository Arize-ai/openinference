from typing import Dict, List

from llama_index.core.base.embeddings.base_sparse import BaseSparseEmbedding
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


class BM25SparseEmbedding(BaseSparseEmbedding):
    model_name: str = "bm25"

    def _tokenize(self, text: str) -> Dict[int, float]:
        tokens = text.lower().split()
        freq: Dict[int, float] = {}
        for token in tokens:
            idx = hash(token) % 30000
            freq[idx] = freq.get(idx, 0.0) + 1.0
        return freq

    def _get_query_embedding(self, query: str) -> Dict[int, float]:
        return self._tokenize(query)

    async def _aget_query_embedding(self, query: str) -> Dict[int, float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Dict[int, float]:
        return self._tokenize(text)

    async def _aget_text_embedding(self, text: str) -> Dict[int, float]:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[Dict[int, float]]:
        return [self._tokenize(t) for t in texts]


embed_model = BM25SparseEmbedding()

texts = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Tokyo is the capital of Japan.",
]


def main() -> None:
    embeddings = embed_model.get_text_embedding_batch(texts)
    for text, emb in zip(texts, embeddings):
        nonzero = {k: v for k, v in emb.items() if v > 0}
        print(f"{text!r}: {len(nonzero)} non-zero dims")


if __name__ == "__main__":
    print("=== Sparse Embedding Example ===")
    main()
