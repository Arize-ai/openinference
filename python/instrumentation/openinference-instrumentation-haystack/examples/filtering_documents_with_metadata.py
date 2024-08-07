from datetime import datetime

from haystack import Document, Pipeline
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from openinference.instrumentation.haystack import HaystackInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Configure HaystackInstrumentor with Phoenix endpoint
endpoint = "http://127.0.0.1:4317"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

HaystackInstrumentor().instrument(tracer_provider=tracer_provider)

documents = [
    Document(
        content="Use pip to install a basic version of Haystack's latest release: pip install farm-haystack. All the core Haystack components live in the haystack repo. But there's also the haystack-extras repo which contains components that are not as widely used, and you need to install them separately.",  # noqa: E501
        meta={"version": 1.15, "date": datetime(2023, 3, 30)},
    ),
    Document(
        content="Use pip to install a basic version of Haystack's latest release: pip install farm-haystack[inference]. All the core Haystack components live in the haystack repo. But there's also the haystack-extras repo which contains components that are not as widely used, and you need to install them separately.",  # noqa: E501
        meta={"version": 1.22, "date": datetime(2023, 11, 7)},
    ),
    Document(
        content="Use pip to install only the Haystack 2.0 code: pip install haystack-ai. The haystack-ai package is built on the main branch which is an unstable beta version, but it's useful if you want to try the new features as soon as they are merged.",  # noqa: E501
        meta={"version": 2.0, "date": datetime(2023, 12, 4)},
    ),
]
document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
document_store.write_documents(documents=documents)


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.add_component(
        instance=InMemoryBM25Retriever(document_store=document_store),
        name="retriever",
    )
    query = "Haystack installation"
    pipeline.run(
        data={
            "retriever": {
                "query": query,
                "filters": {"field": "meta.version", "operator": ">", "value": 1.21},
            }
        }
    )
    query = "Haystack installation"
    pipeline.run(
        data={
            "retriever": {
                "query": query,
                "filters": {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.version", "operator": ">", "value": 1.21},
                        {"field": "meta.date", "operator": ">", "value": datetime(2023, 11, 7)},
                    ],
                },
            }
        }
    )
