"""Integration tests for Amazon Nova model support via invoke_model API.

Cassettes recorded against the real AWS Bedrock API and replayed for CI.
"""

import json

import boto3
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


class TestNovaInvokeModel:
    @pytest.mark.vcr(
        before_record_request=lambda r: r.headers.clear() or r,
    )
    def test_nova_micro_text(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """Basic text response from amazon.nova-micro-v1:0 via invoke_model."""
        model_id = "amazon.nova-micro-v1:0"
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        user_text = "What is 2+2? Reply with just the number."
        request_body = {
            "schemaVersion": "messages-v1",
            "messages": [{"role": "user", "content": [{"text": user_text}]}],
            "inferenceConfig": {"maxTokens": 64, "temperature": 0.1},
        }
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
        )
        response_body = json.loads(response["body"].read())
        assert response_body["output"]["message"]["role"] == "assistant"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.is_ok
        attributes = dict(span.attributes or {})

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
        assert attributes.pop(LLM_MODEL_NAME) == model_id
        assert attributes.pop(INPUT_VALUE) == user_text
        output_value = attributes.pop(OUTPUT_VALUE)
        assert isinstance(output_value, str) and len(output_value) > 0
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)
        assert isinstance(
            invocation_parameters_str := attributes.pop(LLM_INVOCATION_PARAMETERS), str
        )
        assert json.loads(invocation_parameters_str) == {"maxTokens": 64, "temperature": 0.1}
        assert attributes == {}

    @pytest.mark.vcr(
        before_record_request=lambda r: r.headers.clear() or r,
    )
    def test_nova_lite_with_system(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """Nova Lite invoke_model with system prompt and multi-turn messages."""
        model_id = "amazon.nova-lite-v1:0"
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        user_text = "Hello! Who are you?"
        request_body = {
            "schemaVersion": "messages-v1",
            "system": [{"text": "You are a helpful assistant. Keep responses brief."}],
            "messages": [{"role": "user", "content": [{"text": user_text}]}],
            "inferenceConfig": {"maxTokens": 128, "temperature": 0.5},
        }
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
        )
        response_body = json.loads(response["body"].read())
        assert response_body["output"]["message"]["role"] == "assistant"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.is_ok
        attributes = dict(span.attributes or {})

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
        assert attributes.pop(LLM_MODEL_NAME) == model_id
        # INPUT_VALUE comes from the last user message text
        assert attributes.pop(INPUT_VALUE) == user_text
        assert isinstance(attributes.pop(OUTPUT_VALUE), str)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)
        assert isinstance(
            invocation_parameters_str := attributes.pop(LLM_INVOCATION_PARAMETERS), str
        )
        assert json.loads(invocation_parameters_str) == {"maxTokens": 128, "temperature": 0.5}
        assert attributes == {}

    @pytest.mark.vcr(
        before_record_request=lambda r: r.headers.clear() or r,
    )
    def test_nova_converse(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        """Nova model via the converse API — uses unified response format."""
        model_id = "amazon.nova-micro-v1:0"
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        user_text = "What is the capital of France?"
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": user_text}]}],
            inferenceConfig={"maxTokens": 64, "temperature": 0.1},
        )
        assert response["output"]["message"]["role"] == "assistant"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.is_ok
        attributes = dict(span.attributes or {})

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
        assert attributes.pop(LLM_MODEL_NAME) == model_id
        assert attributes.pop(INPUT_VALUE) == user_text
        assert isinstance(attributes.pop(OUTPUT_VALUE), str)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)
        assert isinstance(attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        # converse also sets llm.input_messages and llm.output_messages
        # pop any remaining message-related attributes
        remaining = {k: v for k, v in attributes.items()}
        message_keys = [k for k in remaining if k.startswith("llm.")]
        for k in message_keys:
            attributes.pop(k)
        assert attributes == {}


OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
INPUT_VALUE = SpanAttributes.INPUT_VALUE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
