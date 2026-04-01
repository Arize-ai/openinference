from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

documents = [
    Document(text="Paris is the capital of France."),
    Document(text="Berlin is the capital of Germany."),
    Document(text="Tokyo is the capital of Japan."),
]

if __name__ == "__main__":
    index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever(similarity_top_k=2)
    results = retriever.retrieve("What is the capital of France?")
    for node in results:
        print(node.score, node.text)
