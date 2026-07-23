import json

from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

from openinference.instrumentation.openllmetry._span_processor import _map_generic_span


def test_langchain_retriever_span_maps_query_and_documents() -> None:
    attrs = {
        "traceloop.span.kind": "task",
        "traceloop.entity.name": "VectorStoreRetriever",
        "traceloop.entity.input": json.dumps({"query": "Which VM is low on memory?"}),
        "traceloop.entity.output": json.dumps(
            {
                "documents": [
                    {
                        "page_content": "VM alpha memory 95%",
                        "metadata": {"source": "status"},
                        "id": "doc-1",
                        "score": 0.9,
                    }
                ]
            }
        ),
        "gen_ai.operation.name": "vector_db_retrieve",
        "gen_ai.task.input": json.dumps({"query": "Which VM is low on memory?"}),
        "gen_ai.task.output": json.dumps(
            {
                "documents": [
                    {
                        "page_content": "VM alpha memory 95%",
                        "metadata": {"source": "status"},
                        "id": "doc-1",
                        "score": 0.9,
                    }
                ]
            }
        ),
    }

    mapped = _map_generic_span(attrs, "vector_db_retrieve VectorStoreRetriever")

    assert (
        mapped[SpanAttributes.OPENINFERENCE_SPAN_KIND]
        == OpenInferenceSpanKindValues.RETRIEVER.value
    )
    assert SpanAttributes.TOOL_NAME not in mapped
    assert mapped[SpanAttributes.INPUT_VALUE] == "Which VM is low on memory?"
    assert mapped[SpanAttributes.INPUT_MIME_TYPE] == OpenInferenceMimeTypeValues.TEXT.value

    document_prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0"
    assert mapped[f"{document_prefix}.{DocumentAttributes.DOCUMENT_CONTENT}"] == (
        "VM alpha memory 95%"
    )
    assert mapped[f"{document_prefix}.{DocumentAttributes.DOCUMENT_METADATA}"] == json.dumps(
        {"source": "status"}, separators=(",", ":")
    )
    assert mapped[f"{document_prefix}.{DocumentAttributes.DOCUMENT_ID}"] == "doc-1"
    assert mapped[f"{document_prefix}.{DocumentAttributes.DOCUMENT_SCORE}"] == 0.9


def test_retriever_span_supports_otel_operation_name_and_traceloop_envelopes() -> None:
    mapped = _map_generic_span(
        {
            "traceloop.span.kind": "task",
            "traceloop.entity.input": json.dumps({"query": "fallback query"}),
            "traceloop.entity.output": json.dumps(
                {"documents": [{"content": "fallback document"}]}
            ),
            "gen_ai.operation.name": "retrieval",
        }
    )

    assert (
        mapped[SpanAttributes.OPENINFERENCE_SPAN_KIND]
        == OpenInferenceSpanKindValues.RETRIEVER.value
    )
    assert mapped[SpanAttributes.INPUT_VALUE] == "fallback query"
    assert (
        mapped[f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0.{DocumentAttributes.DOCUMENT_CONTENT}"]
        == "fallback document"
    )
