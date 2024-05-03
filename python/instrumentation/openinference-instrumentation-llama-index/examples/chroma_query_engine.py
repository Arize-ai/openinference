import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from llama_index.vector_stores.chroma import ChromaVectorStore
from openinference.instrumentation import using_attributes
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
trace_api.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

LlamaIndexInstrumentor().instrument()

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("essays")

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
vector_store = ChromaVectorStore(chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)


if __name__ == "__main__":
    query_engine = index.as_query_engine()
    with using_attributes(
        session_id="my-test-session",
        user_id="my-test-user",
        metadata={
            "test-int": 1,
            "test-str": "string",
            "test-list": [1, 2, 3],
            "test-dict": {
                "key-1": "val-1",
                "key-2": "val-2",
            },
        },
        tags=["tag-1", "tag-2"],
    ):
        response = query_engine.query("What did the author do growing up?")
        print(response)
