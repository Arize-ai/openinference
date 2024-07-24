from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from openinference.instrumentation.haystack import HaystackInstrumentor  # CHANGE
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Configure HaystackInstrumentor with Phoenix endpoint
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

HaystackInstrumentor().instrument(tracer_provider=tracer_provider)

# Configure document store and load dataset
document_store = InMemoryDocumentStore()

dataset = load_dataset("bilgeyucel/seven-wonders", split="train")

docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

# Configure document embedder and store documents from dataset
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()
docs_with_embeddings = doc_embedder.run(docs)

document_store.write_documents(docs_with_embeddings["documents"])

# Configure text embedder (prompt)
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

# Configure retriever
retriever = InMemoryEmbeddingRetriever(document_store)

# Set up template for prompt + docs
template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

# Configure prompt builder
prompt_builder = PromptBuilder(template=template)

# Configure generator
generator = OpenAIGenerator(model="gpt-3.5-turbo")

# Instantiate pipeline
basic_rag_pipeline = Pipeline()

# Add components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", generator)

# Connect necessary pipeline components to each other
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")

# Define the question and run the pipeline
if __name__ == "__main__":
    q = "What is the location of the Hanging Gardens of Babylon?"
    basic_rag_pipeline.run({"text_embedder": {"text": q}, "prompt_builder": {"question": q}})
