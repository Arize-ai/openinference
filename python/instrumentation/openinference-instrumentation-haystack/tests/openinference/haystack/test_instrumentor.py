from typing import Any, Dict, Generator, List, Optional, Union

import pytest
from haystack import Document
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.core.pipeline.pipeline import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils.auth import Secret
from openinference.instrumentation.haystack import HaystackInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def fake_run(
    self: Any, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, List[Union[str, Dict[str, Any]]]]:
    return {
        "replies": ["sorry, i have zero clue"],
        "meta": [
            {
                "model": "parth",
                "usage": {"completion_tokens": 10, "prompt_tokens": 5, "total_tokens": 15},
                "awesome_attribute": True,
                "another_attribute": 7,
            }
        ],
    }


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def setup_haystack_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    HaystackInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    HaystackInstrumentor().uninstrument()


def test_haystack_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
) -> None:
    # Configure document store and load dataset
    document_store = InMemoryDocumentStore()

    dataset = [
        {
            "content": "The first wonder of the world is the Great Pyramid of Giza",
            "meta": {"id": 1},
        },
        {
            "content": "The second wonder of the world is the Hanging Gardens of Babylon",
            "meta": {"id": 2},
        },
        {
            "content": "The third wonder of the world is the Statue of Zeus at Olympia",
            "meta": {"id": 3},
        },
    ]

    docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

    # Configure document embedder and store documents from dataset
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    doc_embedder.warm_up()
    docs_with_embeddings = doc_embedder.run(docs)

    document_store.write_documents(docs_with_embeddings["documents"])

    # Configure text embedder (prompt)
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

    # Configure retriever
    retriever = InMemoryEmbeddingRetriever(document_store)

    template = """
    Given the following information, answer the question.
    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}
    Question: {{question}}
    Answer:
    """

    prompt_builder = PromptBuilder(template=template)
    llm = OpenAIGenerator(api_key=Secret.from_token("TOTALLY_REAL_API_KEY"))
    llm.run = fake_run.__get__(llm, OpenAIGenerator)

    basic_rag_pipeline = Pipeline()
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("llm", llm)

    # Connect necessary pipeline components to each other
    basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
    basic_rag_pipeline.connect("prompt_builder", "llm")

    q = "What's Natural Language Processing? Be brief."

    basic_rag_pipeline.run({"text_embedder": {"text": q}, "prompt_builder": {"question": q}})

    spans = in_memory_span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "SentenceTransformersTextEmbedder",
        "InMemoryEmbeddingRetriever",
        "PromptBuilder",
        "OpenAIGenerator",
        "Pipeline",
    ]

    assert [
        span.attributes.get("openinference.span.kind") for span in spans if span and span.attributes
    ] == [
        "EMBEDDING",
        "RETRIEVER",
        "CHAIN",
        "LLM",
        "CHAIN",
    ]


def test_haystack_uninstrumentation(
    tracer_provider: TracerProvider,
) -> None:
    # Instrumenting Haystack
    HaystackInstrumentor().instrument(tracer_provider=tracer_provider)

    # Ensure methods are wrapped
    assert hasattr(Pipeline.run, "__wrapped__")
    assert hasattr(Pipeline._run_component, "__wrapped__")

    # Uninstrumenting Haystack
    HaystackInstrumentor().uninstrument()

    # Ensure methods are not wrapped
    assert not hasattr(Pipeline.run, "__wrapped__")
    assert not hasattr(Pipeline._run_component, "__wrapped__")
