import chromadb
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
span_exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:6006/v1/traces")
span_processor = SimpleSpanProcessor(span_exporter=span_exporter)
tracer_provider.add_span_processor(span_processor=span_processor)
trace_api.set_tracer_provider(tracer_provider=tracer_provider)

LlamaIndexInstrumentor().instrument()
OpenAIInstrumentor().instrument()

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("essays")

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)


if __name__ == "__main__":
    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do growing up?")
    print(response)
