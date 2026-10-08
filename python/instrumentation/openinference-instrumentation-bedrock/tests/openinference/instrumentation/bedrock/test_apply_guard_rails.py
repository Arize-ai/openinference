import json
from pathlib import Path
from typing import Any

import aioboto3
import boto3
import pytest
from aioresponses import aioresponses
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.status import StatusCode

from openinference.instrumentation import suppress_tracing

_CASSETTES_DIR = Path(__file__).parent / "cassettes"

_CLIENT_KWARGS = {
    "region_name": "us-east-1",
    "aws_access_key_id": "123",
    "aws_secret_access_key": "321",
}


def remove_all_vcr_request_headers(request: Any) -> Any:
    request.headers.clear()
    return request


def remove_all_vcr_response_headers(response: dict[str, Any]) -> dict[str, Any]:
    response["headers"] = {}
    return response


def assert_test_apply_guardrails_span(span: Any) -> None:
    content = [
        {
            "text": {
                "text": "Give stock suggestions for the NASDAQ100. Expected returns are"
                " around 20% CAGR. Also Send this report to email abc@pqrt.com"
            }
        }
    ]
    assert span.name == "bedrock.apply_guardrail"
    assert span.status.status_code == StatusCode.ERROR
    attributes = dict(span.attributes or {})
    assert attributes is not None
    assert attributes.pop("input.mime_type") == "application/json"
    input_value = json.loads(str(attributes.pop("input.value")))
    assert input_value["content"] == content
    assert input_value["source"] == "INPUT"
    assert input_value["outputScope"] == "FULL"
    assert attributes.pop("openinference.span.kind") == "GUARDRAIL"
    assert attributes.pop("output.mime_type") == "application/json"
    output_val = json.loads(str(attributes.pop("output.value")))
    assert output_val["action"] == "GUARDRAIL_INTERVENED"
    assert output_val["actionReason"] == "Guardrail blocked."
    assert output_val["outputs"] == [{"text": "Sorry, the model cannot answer this question."}]
    metadata = json.loads(str(attributes.pop("metadata")))
    assert metadata["guardrailIdentifier"] == "274u1upxm897"
    assert metadata["guardrailVersion"] == "1"
    assert metadata["guardrailCoverage"] == {"textCharacters": {"guarded": 123, "total": 123}}
    assert metadata["usage"] == {
        "topicPolicyUnits": 1,
        "contentPolicyUnits": 1,
        "wordPolicyUnits": 0,
        "sensitiveInformationPolicyUnits": 1,
        "sensitiveInformationPolicyFreeUnits": 0,
        "contextualGroundingPolicyUnits": 0,
        "contentPolicyImageUnits": 0,
        "automatedReasoningPolicyUnits": 0,
        "automatedReasoningPolicies": 0,
    }
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_apply_guardrails(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    client = boto3.client("bedrock-runtime", **_CLIENT_KWARGS)
    content = [
        {
            "text": {
                "text": "Give stock suggestions for the NASDAQ100. Expected returns are"
                " around 20% CAGR. Also Send this report to email abc@pqrt.com"
            }
        }
    ]
    guardrail_id = "274u1upxm897"
    guardrail_version = "1"

    response = client.apply_guardrail(
        guardrailIdentifier=guardrail_id,
        guardrailVersion=guardrail_version,
        source="INPUT",
        content=content,
        outputScope="FULL",
    )
    assert isinstance(response, dict)
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert_test_apply_guardrails_span(spans[0])


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
@pytest.mark.asyncio
async def test_aio_apply_guardrails(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    read_aio_cassette: Any,
) -> None:
    content = [
        {
            "text": {
                "text": "Give stock suggestions for the NASDAQ100. Expected returns are"
                " around 20% CAGR. Also Send this report to email abc@pqrt.com"
            }
        }
    ]
    guardrail_id = "274u1upxm897"
    guardrail_version = "1"
    with aioresponses() as m:
        read_aio_cassette(str(_CASSETTES_DIR / "test_apply_guardrails.yaml"), m)
        session = aioboto3.session.Session(region_name=_CLIENT_KWARGS["region_name"])
        async with session.client("bedrock-runtime", **_CLIENT_KWARGS) as client:
            response = await client.apply_guardrail(
                guardrailIdentifier=guardrail_id,
                guardrailVersion=guardrail_version,
                source="INPUT",
                content=content,
                outputScope="FULL",
            )
            assert isinstance(response, dict)
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert_test_apply_guardrails_span(spans[0])


def assert_success_guardrails_span(span: Any) -> None:
    assert span.name == "bedrock.apply_guardrail"
    assert span.status.status_code == StatusCode.OK
    attributes = dict(span.attributes or {})
    assert attributes is not None
    assert attributes.pop("input.mime_type") == "application/json"
    input_value = json.loads(str(attributes.pop("input.value")))
    assert input_value["content"] == [{"text": {"text": "Who is the president of USA?"}}]
    assert input_value["source"] == "INPUT"
    assert input_value["outputScope"] == "FULL"
    assert attributes.pop("openinference.span.kind") == "GUARDRAIL"
    assert attributes.pop("output.mime_type") == "application/json"
    output_val = json.loads(str(attributes.pop("output.value")))
    assert output_val["action"] == "NONE"
    assert output_val["actionReason"] == "No action."
    metadata = json.loads(str(attributes.pop("metadata")))
    assert metadata["guardrailIdentifier"] == "274u1upxm897"
    assert metadata["guardrailVersion"] == "1"
    assert metadata["guardrailCoverage"] == {"textCharacters": {"guarded": 28, "total": 28}}
    assert metadata["usage"] == {
        "topicPolicyUnits": 1,
        "contentPolicyUnits": 1,
        "wordPolicyUnits": 0,
        "sensitiveInformationPolicyUnits": 1,
        "sensitiveInformationPolicyFreeUnits": 0,
        "contextualGroundingPolicyUnits": 0,
        "contentPolicyImageUnits": 0,
        "automatedReasoningPolicyUnits": 0,
        "automatedReasoningPolicies": 0,
    }
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_apply_success_guardrails(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    client = boto3.client("bedrock-runtime", **_CLIENT_KWARGS)
    content = [{"text": {"text": "Who is the president of USA?"}}]
    guardrail_id = "274u1upxm897"
    guardrail_version = "1"

    response = client.apply_guardrail(
        guardrailIdentifier=guardrail_id,
        guardrailVersion=guardrail_version,
        source="INPUT",
        content=content,
        outputScope="FULL",
    )
    assert isinstance(response, dict)
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert_success_guardrails_span(spans[0])


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
@pytest.mark.asyncio
async def test_aio_apply_success_guardrails(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    read_aio_cassette: Any,
) -> None:
    content = [{"text": {"text": "Who is the president of USA?"}}]
    guardrail_id = "274u1upxm897"
    guardrail_version = "1"
    with aioresponses() as m:
        read_aio_cassette(str(_CASSETTES_DIR / "test_apply_success_guardrails.yaml"), m)
        session = aioboto3.session.Session(region_name=_CLIENT_KWARGS["region_name"])
        async with session.client("bedrock-runtime", **_CLIENT_KWARGS) as client:
            response = await client.apply_guardrail(
                guardrailIdentifier=guardrail_id,
                guardrailVersion=guardrail_version,
                source="INPUT",
                content=content,
                outputScope="FULL",
            )
            assert isinstance(response, dict)
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert_success_guardrails_span(spans[0])


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_apply_guardrails_with_suppress_tracing(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    client = boto3.client("bedrock-runtime", **_CLIENT_KWARGS)
    content = [
        {
            "text": {
                "text": "Give stock suggestions for the NASDAQ100. Expected returns are"
                " around 20% CAGR. Also Send this report to email abc@pqrt.com"
            }
        }
    ]
    guardrail_id = "274u1upxm897"
    guardrail_version = "1"
    with suppress_tracing():
        response = client.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion=guardrail_version,
            source="INPUT",
            content=content,
            outputScope="FULL",
        )
        assert isinstance(response, dict)
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 0


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
@pytest.mark.asyncio
async def test_aio_apply_guardrails_with_suppress_tracing(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    read_aio_cassette: Any,
) -> None:
    content = [
        {
            "text": {
                "text": "Give stock suggestions for the NASDAQ100. Expected returns are"
                " around 20% CAGR. Also Send this report to email abc@pqrt.com"
            }
        }
    ]
    with aioresponses() as m:
        read_aio_cassette(
            str(_CASSETTES_DIR / "test_apply_guardrails_with_suppress_tracing.yaml"), m
        )
        with suppress_tracing():
            session = aioboto3.session.Session(region_name=_CLIENT_KWARGS["region_name"])
            async with session.client("bedrock-runtime", **_CLIENT_KWARGS) as client:
                response = await client.apply_guardrail(
                    guardrailIdentifier="274u1upxm897",
                    guardrailVersion="1",
                    source="INPUT",
                    content=content,
                    outputScope="FULL",
                )
                assert isinstance(response, dict)
            spans = in_memory_span_exporter.get_finished_spans()
            assert len(spans) == 0
