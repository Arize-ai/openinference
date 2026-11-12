"""Tests that retriever spans record the LangChain ``Document.id``.

``langchain_core.documents.Document`` has an optional ``id`` field (added in
langchain-core 0.2.11) that vector stores fill in. It must be recorded as
``retrieval.documents.N.document.id`` so retrieved sources can be identified.
"""

import json
from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
INPUT_VALUE = SpanAttributes.INPUT_VALUE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
METADATA = SpanAttributes.METADATA
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
    assert (
        attributes.pop(OPENINFERENCE_SPAN_KIND, None) == OpenInferenceSpanKindValues.RETRIEVER.value
    )
    assert attributes.pop(INPUT_VALUE, None) == "query"
    assert isinstance(output_value := attributes.pop(OUTPUT_VALUE, None), str)
    assert [(d["id"], d["page_content"]) for d in json.loads(output_value)["documents"]] == [
        ("doc-1", "first"),
        (None, "second"),
    ]
    assert attributes.pop(OUTPUT_MIME_TYPE, None) == OpenInferenceMimeTypeValues.JSON.value
    assert isinstance(metadata := attributes.pop(METADATA, None), str)
    assert json.loads(metadata) == {"ls_retriever_name": "_static"}
    assert attributes.pop(f"{RETRIEVAL_DOCUMENTS}.0.{DOCUMENT_ID}", None) == "doc-1"
    assert attributes.pop(f"{RETRIEVAL_DOCUMENTS}.0.{DOCUMENT_CONTENT}", None) == "first"
    assert attributes.pop(f"{RETRIEVAL_DOCUMENTS}.1.{DOCUMENT_CONTENT}", None) == "second"
    assert attributes == {}
