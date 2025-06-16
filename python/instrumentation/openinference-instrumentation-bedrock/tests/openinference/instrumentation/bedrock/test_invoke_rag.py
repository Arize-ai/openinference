from typing import Any, Dict

import boto3
import pytest
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def starts_with(left_value: Any, right_value: str) -> bool:
    return str(left_value).startswith(right_value)


def remove_all_vcr_request_headers(request: Any) -> Any:
    """
    Removes all request headers.

    Example:
    ```
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_request_headers
    )
    def test_openai() -> None:
        # make request to OpenAI
    """
    request.headers.clear()
    return request


def remove_all_vcr_response_headers(response: dict[str, Any]) -> dict[str, Any]:
    """
    Removes all response headers.

    Example:
    ```
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_response_headers
    )
    def test_openai() -> None:
        # make request to OpenAI
    """
    response["headers"] = {}
    return response


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_retrieve(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name="ap-south-1",
        aws_access_key_id="123",
        aws_secret_access_key="321",
    )
    attributes = dict(
        knowledgeBaseId="SSGLURQ9A5", retrievalQuery={"text": "What is task Decomposition?"}
    )
    response = client.retrieve(**attributes)
    assert isinstance(response, dict)
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs: Dict[str, Any] = dict(spans[0].attributes or {})
    assert attrs.pop("input.mime_type") == "text/plain"
    assert attrs.pop("input.value").startswith("What is task")

    invocation = attrs.pop("llm.invocation_parameters")
    assert '"knowledgeBaseId": "SSGLURQ9A5"' in invocation

    assert attrs.pop("openinference.span.kind") == "RETRIEVER"

    for i in range(5):
        prefix = f"retrieval.documents.{i}.document"

        content = attrs.pop(f"{prefix}.content")
        assert isinstance(content, str)
        assert content[:15].strip() != ""  # crude check for non-empty

        metadata = attrs.pop(f"{prefix}.metadata")
        assert '"customDocumentLocation": {"id": "2222"}' in metadata

        score = attrs.pop(f"{prefix}.score")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    # At the end, ensure no unexpected fields
    assert not attrs, f"Unexpected extra attributes: {attrs}"


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_retrieve_and_generate_with_knowledge_base(
    tracer_provider: trace_sdk.TracerProvider, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name="ap-south-1",
        aws_access_key_id="123",
        aws_secret_access_key="321",
    )
    attributes = {
        "input": {"text": "What is Task Decomposition?"},
        "retrieveAndGenerateConfiguration": {
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": "SSGLURQ9A5",
                "modelArn": "anthropic.claude-3-haiku-20240307-v1:0",
            },
            "type": "KNOWLEDGE_BASE",
        },
    }
    response = client.retrieve_and_generate(**attributes)
    assert isinstance(response, dict)
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs: Dict[str, Any] = dict(spans[0].attributes or {})
    assert attrs.pop("input.mime_type") == "text/plain"
    assert attrs.pop("input.value").startswith("What is Task")

    invocation = attrs.pop("llm.invocation_parameters")
    assert '"retrieveAndGenerateConfiguration"' in invocation
    assert '"knowledgeBaseId": "SSGLURQ9A5"' in invocation

    assert attrs.pop("llm.model_name") == "anthropic.claude-3-haiku-20240307-v1:0"
    assert attrs.pop("openinference.span.kind") == "RETRIEVER"
    assert attrs.pop("output.mime_type") == "text/plain"

    output_val = attrs.pop("output.value")
    assert output_val.startswith("Task Decomposition is a technique")
    assert "Chain of Thought" in output_val
    assert "Tree of Thoughts" in output_val

    for i in range(2):
        prefix = f"retrieval.documents.{i}.document"

        content = attrs.pop(f"{prefix}.content")
        assert isinstance(content, str)
        assert "Task Decomposition" in content

        metadata = attrs.pop(f"{prefix}.metadata")
        assert '"customDocumentLocation": {"id": "2222"}' in metadata
        assert '"x-amz-bedrock-kb-data-source-id": "VYV3J5D9O6"' in metadata

    # Final assertion: no unexpected attributes remain
    assert not attrs, f"Unexpected attributes found: {attrs}"


def validate_rag_span_attributes(memory_span_exporter: InMemorySpanExporter) -> None:
    spans = memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs: Dict[str, Any] = dict(spans[0].attributes or {})
    assert attrs.pop("input.mime_type") == "text/plain"
    assert attrs.pop("input.value") == "What is Telos?"
    assert attrs.pop("llm.model_name") == "anthropic.claude-3-haiku-20240307-v1:0"
    assert attrs.pop("openinference.span.kind") == "RETRIEVER"
    assert attrs.pop("output.mime_type") == "text/plain"

    output = attrs.pop("output.value")
    assert "Telos is a knowledge representation language" in output
    assert "Telos treats attributes as first-class citizens" in output
    assert "Telos propositions are organized along three dimensions" in output
    assert "history time and belief time" in output
    assert "assertion language for expressing deductive rules" in output

    # Validate retrieval documents
    for i in range(9):
        content_key = f"retrieval.documents.{i}.document.content"
        metadata_key = f"retrieval.documents.{i}.document.metadata"
        if content_key in attrs:
            content = attrs.pop(content_key)
            assert content is not None
        if metadata_key in attrs:
            metadata = attrs.pop(metadata_key)
            assert "s3://bedrock-az-kb/knowledge_bases/VLDBJ96.pdf" in metadata

    # Validate invocation parameters
    invocation = attrs.pop("llm.invocation_parameters")
    assert '"sourceType": "S3"' in invocation
    assert '"modelArn": "anthropic.claude-3-haiku-20240307-v1:0"' in invocation

    # Final assertion: no unexpected attributes remain
    assert not attrs, f"Unexpected attributes found: {attrs}"


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_retrieve_and_generate_with_external_s3_source(
    tracer_provider: trace_sdk.TracerProvider, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name="ap-south-1",
        aws_access_key_id="123",
        aws_secret_access_key="321",
    )
    response = client.retrieve_and_generate(
        input={"text": "What is Telos?"},
        retrieveAndGenerateConfiguration={
            "externalSourcesConfiguration": {
                "sources": [
                    {
                        "s3Location": {"uri": "s3://bedrock-az-kb/knowledge_bases/VLDBJ96.pdf"},
                        "sourceType": "S3",
                    }
                ],
                "modelArn": "anthropic.claude-3-haiku-20240307-v1:0",
            },
            "type": "EXTERNAL_SOURCES",
        },
    )
    assert isinstance(response, dict)
    validate_rag_span_attributes(in_memory_span_exporter)


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_retrieve_and_generate_stream(
    tracer_provider: trace_sdk.TracerProvider, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name="ap-south-1",
        aws_access_key_id="123",
        aws_secret_access_key="321",
    )
    response_stream = client.retrieve_and_generate_stream(
        input={"text": "What is Telos?"},
        retrieveAndGenerateConfiguration={
            "externalSourcesConfiguration": {
                "sources": [
                    {
                        "s3Location": {"uri": "s3://bedrock-az-kb/knowledge_bases/VLDBJ96.pdf"},
                        "sourceType": "S3",
                    }
                ],
                "modelArn": "anthropic.claude-3-haiku-20240307-v1:0",
            },
            "type": "EXTERNAL_SOURCES",
        },
    )
    response = list(response_stream["stream"])  # Collect all streamed responses
    assert isinstance(response, list)
    validate_rag_span_attributes(in_memory_span_exporter)
