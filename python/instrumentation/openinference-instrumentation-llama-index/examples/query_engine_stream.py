import tempfile
from urllib.request import urlretrieve

import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("essays")
vector_store = ChromaVectorStore(chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
with tempfile.NamedTemporaryFile() as tf:
    urlretrieve(
        "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt",
        tf.name,
    )
    documents = SimpleDirectoryReader(input_files=[tf.name]).load_data()
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
query_engine = index.as_query_engine(streaming=True)
Settings.llm = OpenAI(model="gpt-3.5-turbo")

if __name__ == "__main__":
    response = query_engine.query("What did the author do growing up?")
    response.print_response_stream()
