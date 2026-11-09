"""Tests that retriever spans record the LangChain ``Document.id``.

``langchain_core.documents.Document`` has an optional ``id`` field (added in
langchain-core 0.2.11) that vector stores fill in. It must be recorded as
``retrieval.documents.N.document.id`` so retrieved sources can be identified.
"""

from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.semconv.trace import DocumentAttributes, SpanAttributes

RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT


class _StaticRetriever(BaseRetriever):
    documents: List[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.documents


def test_retriever_span_records_document_id(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    retriever = _StaticRetriever(
        documents=[
            Document(id="doc-1", page_content="first"),
            Document(page_content="second"),  # no id: nothing recorded
        ]
    )
    retriever.invoke("query")

    (span,) = in_memory_span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes[f"{RETRIEVAL_DOCUMENTS}.0.{DOCUMENT_ID}"] == "doc-1"
    assert attributes[f"{RETRIEVAL_DOCUMENTS}.0.{DOCUMENT_CONTENT}"] == "first"
    assert f"{RETRIEVAL_DOCUMENTS}.1.{DOCUMENT_ID}" not in attributes
    assert attributes[f"{RETRIEVAL_DOCUMENTS}.1.{DOCUMENT_CONTENT}"] == "second"
