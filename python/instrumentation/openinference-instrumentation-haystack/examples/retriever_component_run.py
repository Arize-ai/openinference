from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.haystack import HaystackInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

HaystackInstrumentor().instrument(tracer_provider=tracer_provider)

document_store = InMemoryDocumentStore()
documents = [
    Document(content="There are over 7,000 languages spoken around the world today."),
    Document(
        content="Elephants have been observed to behave in a way that indicates "
        "a high level of self-awareness, such as recognizing themselves "
        "in mirrors."
    ),
    Document(
        content="In certain parts of the world, like the Maldives, Puerto Rico, "
        "and San Diego, you can witness the phenomenon of bioluminescent "
        "waves."
    ),
]
document_store.write_documents(documents=documents)

retriever = InMemoryBM25Retriever(document_store=document_store)
print(retriever.run(query="How many languages are spoken around the world today?"))
