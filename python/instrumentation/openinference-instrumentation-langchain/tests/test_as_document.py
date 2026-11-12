from langchain_core.documents import Document
from openinference.semconv.trace import DocumentAttributes

from openinference.instrumentation.langchain._tracer import _as_document

DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT


def test_as_document_records_id() -> None:
    attributes = dict(_as_document(Document(id="m1", page_content="hello")))
    assert attributes[DOCUMENT_ID] == "m1"
    assert attributes[DOCUMENT_CONTENT] == "hello"


def test_as_document_without_id() -> None:
    attributes = dict(_as_document(Document(page_content="hello")))
    assert DOCUMENT_ID not in attributes
    assert attributes[DOCUMENT_CONTENT] == "hello"
