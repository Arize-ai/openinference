from pathlib import Path
from typing import Any, Callable, Dict

import aioboto3
import boto3
import pytest
from aioresponses import aioresponses
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_CASSETTES_DIR = Path(__file__).parent / "cassettes"
_CLIENT_KWARGS = {
    "region_name": "ap-south-1",
    "aws_access_key_id": "123",
    "aws_secret_access_key": "321",
}

_RETRIEVE_ATTRIBUTES = dict(
    knowledgeBaseId="SSGLURQ9A5",
    retrievalQuery={"text": "What is task Decomposition?"},
)
_RETRIEVE_AND_GENERATE_KB_ATTRIBUTES = {
    "input": {"text": "What is Task Decomposition?"},
    "retrieveAndGenerateConfiguration": {
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": "SSGLURQ9A5",
            "modelArn": "anthropic.claude-3-haiku-20240307-v1:0",
        },
        "type": "KNOWLEDGE_BASE",
    },
}
_EXTERNAL_S3_RAG_INPUT = {"text": "What is Telos?"}
_EXTERNAL_S3_RAG_CONFIG = {
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
}


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


def _assert_retrieve_span_attributes(
    attrs: Dict[str, Any],
    *,
    num_docs: int = 5,
    doc_id_in_metadata: str = "2222",
) -> None:
    assert attrs.pop("input.mime_type") == "text/plain"
    assert attrs.pop("input.value").startswith("What is task")
    invocation = attrs.pop("llm.invocation_parameters")
    assert '"knowledgeBaseId": "SSGLURQ9A5"' in invocation
    assert attrs.pop("openinference.span.kind") == "RETRIEVER"
    for i in range(num_docs):
        prefix = f"retrieval.documents.{i}.document"
        content = attrs.pop(f"{prefix}.content")
        assert isinstance(content, str)
        assert content[:15].strip() != ""
        metadata = attrs.pop(f"{prefix}.metadata")
        assert f'"customDocumentLocation": {{"id": "{doc_id_in_metadata}"}}' in metadata
        score = attrs.pop(f"{prefix}.score")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    assert not attrs, f"Unexpected extra attributes: {attrs}"


def _assert_retrieve_and_generate_kb_span_attributes(attrs: Dict[str, Any]) -> None:
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
    assert not attrs, f"Unexpected attributes found: {attrs}"


def _assert_single_rag_span(
    in_memory_span_exporter: InMemorySpanExporter,
    assert_attrs: Callable[[Dict[str, Any]], None],
) -> None:
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert_attrs(dict(spans[0].attributes or {}))


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_retrieve(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    client = boto3.client("bedrock-agent-runtime", **_CLIENT_KWARGS)
    response = client.retrieve(**_RETRIEVE_ATTRIBUTES)
    assert isinstance(response, dict)
    _assert_single_rag_span(in_memory_span_exporter, _assert_retrieve_span_attributes)


@pytest.mark.order(after="test_retrieve")
async def test_async_retrieve(
    in_memory_span_exporter: InMemorySpanExporter,
    read_aio_cassette: Any,
) -> None:
    """Async version of test_retrieve; same cassette and assertions."""
    with aioresponses() as m:
        read_aio_cassette(str(_CASSETTES_DIR / "test_retrieve.yaml"), m)
        session = aioboto3.session.Session(region_name=_CLIENT_KWARGS["region_name"])
        async with session.client("bedrock-agent-runtime", **_CLIENT_KWARGS) as client:
            response = await client.retrieve(**_RETRIEVE_ATTRIBUTES)
    assert isinstance(response, dict)
    _assert_single_rag_span(in_memory_span_exporter, _assert_retrieve_span_attributes)


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_retrieve_and_generate_with_knowledge_base(
    tracer_provider: trace_sdk.TracerProvider, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    client = boto3.client("bedrock-agent-runtime", **_CLIENT_KWARGS)
    response = client.retrieve_and_generate(**_RETRIEVE_AND_GENERATE_KB_ATTRIBUTES)
    assert isinstance(response, dict)
    _assert_single_rag_span(
        in_memory_span_exporter, _assert_retrieve_and_generate_kb_span_attributes
    )


@pytest.mark.order(after="test_retrieve_and_generate_with_knowledge_base")
async def test_async_retrieve_and_generate_with_knowledge_base(
    in_memory_span_exporter: InMemorySpanExporter,
    read_aio_cassette: Any,
) -> None:
    """Async version of test_retrieve_and_generate_with_knowledge_base; same cassette."""
    with aioresponses() as m:
        read_aio_cassette(
            str(_CASSETTES_DIR / "test_retrieve_and_generate_with_knowledge_base.yaml"),
            m,
        )
        session = aioboto3.session.Session(region_name=_CLIENT_KWARGS["region_name"])
        async with session.client("bedrock-agent-runtime", **_CLIENT_KWARGS) as client:
            response = await client.retrieve_and_generate(**_RETRIEVE_AND_GENERATE_KB_ATTRIBUTES)
    assert isinstance(response, dict)
    _assert_single_rag_span(
        in_memory_span_exporter, _assert_retrieve_and_generate_kb_span_attributes
    )


def _assert_external_s3_rag_span_attributes(attrs: Dict[str, Any]) -> None:
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


def validate_rag_span_attributes(memory_span_exporter: InMemorySpanExporter) -> None:
    _assert_single_rag_span(memory_span_exporter, _assert_external_s3_rag_span_attributes)


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_retrieve_and_generate_with_external_s3_source(
    tracer_provider: trace_sdk.TracerProvider, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    client = boto3.client("bedrock-agent-runtime", **_CLIENT_KWARGS)
    response = client.retrieve_and_generate(
        input=_EXTERNAL_S3_RAG_INPUT,
        retrieveAndGenerateConfiguration=_EXTERNAL_S3_RAG_CONFIG,
    )
    assert isinstance(response, dict)
    validate_rag_span_attributes(in_memory_span_exporter)


@pytest.mark.order(after="test_retrieve_and_generate_with_external_s3_source")
async def test_async_retrieve_and_generate_with_external_s3_source(
    in_memory_span_exporter: InMemorySpanExporter,
    read_aio_cassette: Any,
) -> None:
    """Async version of test_retrieve_and_generate_with_external_s3_source; same cassette."""
    with aioresponses() as m:
        read_aio_cassette(
            str(_CASSETTES_DIR / "test_retrieve_and_generate_with_external_s3_source.yaml"),
            m,
        )
        session = aioboto3.session.Session(region_name=_CLIENT_KWARGS["region_name"])
        async with session.client("bedrock-agent-runtime", **_CLIENT_KWARGS) as client:
            response = await client.retrieve_and_generate(
                input=_EXTERNAL_S3_RAG_INPUT,
                retrieveAndGenerateConfiguration=_EXTERNAL_S3_RAG_CONFIG,
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
    client = boto3.client("bedrock-agent-runtime", **_CLIENT_KWARGS)
    response_stream = client.retrieve_and_generate_stream(
        input=_EXTERNAL_S3_RAG_INPUT,
        retrieveAndGenerateConfiguration=_EXTERNAL_S3_RAG_CONFIG,
    )
    response = list(response_stream["stream"])
    assert isinstance(response, list)
    validate_rag_span_attributes(in_memory_span_exporter)


@pytest.mark.order(after="test_retrieve_and_generate_stream")
async def test_async_retrieve_and_generate_stream(
    in_memory_span_exporter: InMemorySpanExporter,
    read_aio_cassette: Any,
) -> None:
    """Async version of test_retrieve_and_generate_stream; same cassette."""
    with aioresponses() as m:
        read_aio_cassette(
            str(_CASSETTES_DIR / "test_retrieve_and_generate_stream.yaml"),
            m,
        )
        session = aioboto3.session.Session(region_name=_CLIENT_KWARGS["region_name"])
        async with session.client("bedrock-agent-runtime", **_CLIENT_KWARGS) as client:
            response_stream = await client.retrieve_and_generate_stream(
                input=_EXTERNAL_S3_RAG_INPUT,
                retrieveAndGenerateConfiguration=_EXTERNAL_S3_RAG_CONFIG,
            )
            events = []
            async for event in response_stream["stream"]:
                events.append(event)
    assert len(events) > 0
    validate_rag_span_attributes(in_memory_span_exporter)
