"""Integration tests for OpenAI models (gpt-oss, GPT-5.x, GPT-6) via the invoke_model APIs.

These models take and return an OpenAI Chat Completions body on InvokeModel.
Cassettes recorded against the real AWS Bedrock API and replayed for CI.
"""

import json
from typing import Any
from unittest.mock import MagicMock

import boto3
import pytest
from botocore.eventstream import EventStream
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.bedrock._wrappers import _InvokeModelWithResponseStream
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

_MODEL_ID = "us.openai.gpt-6-sol"
_CLIENT_KWARGS = {
    "region_name": "us-east-1",
    "aws_access_key_id": "123",
    "aws_secret_access_key": "321",
}


class TestOpenAIInvokeModel:
    @pytest.mark.vcr(
        before_record_request=lambda r: r.headers.clear() or r,
    )
    def test_invoke_model(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        messages = [
            {"role": "system", "content": "Answer in one word."},
            {"role": "user", "content": "say pong"},
        ]
        client = boto3.client("bedrock-runtime", **_CLIENT_KWARGS)
        response = client.invoke_model(
            modelId=_MODEL_ID,
            body=json.dumps({"messages": messages, "max_completion_tokens": 256}),
        )
        response_body = json.loads(response["body"].read())

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.is_ok
        attributes = dict(span.attributes or {})

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
        assert attributes.pop(LLM_MODEL_NAME) == _MODEL_ID
        assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "aws"
        assert attributes.pop(INPUT_VALUE) == json.dumps(messages)
        assert attributes.pop(INPUT_MIME_TYPE) == "application/json"
        assert json.loads(str(attributes.pop(LLM_INVOCATION_PARAMETERS))) == {
            "max_completion_tokens": 256
        }
        assert attributes.pop(OUTPUT_VALUE) == response_body["choices"][0]["message"]["content"]
        assert attributes.pop(LLM_FINISH_REASON) == "stop"
        usage = response_body["usage"]
        assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == usage["prompt_tokens"]
        assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == usage["completion_tokens"]
        assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == usage["total_tokens"]
        assert attributes == {}


class TestOpenAIInvokeModelWithResponseStream:
    @pytest.mark.vcr(
        before_record_request=lambda r: r.headers.clear() or r,
    )
    def test_streaming(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        messages = [{"role": "user", "content": "count 1 to 5"}]
        request_body = {
            "messages": messages,
            "max_completion_tokens": 256,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        client = boto3.client("bedrock-runtime", **_CLIENT_KWARGS)
        response = client.invoke_model_with_response_stream(
            modelId=_MODEL_ID,
            body=json.dumps(request_body),
        )
        text = ""
        usage: dict[str, Any] = {}
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            for choice in chunk["choices"]:
                text += choice["delta"].get("content") or ""
            usage = chunk.get("usage") or usage
        assert text

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.is_ok
        attributes = dict(span.attributes or {})

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
        assert attributes.pop(LLM_MODEL_NAME) == _MODEL_ID
        assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "aws"
        assert attributes.pop(INPUT_VALUE) == json.dumps(messages)
        assert attributes.pop(INPUT_MIME_TYPE) == "application/json"
        assert json.loads(str(attributes.pop(LLM_INVOCATION_PARAMETERS))) == {
            "max_completion_tokens": 256,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        assert attributes.pop(OUTPUT_VALUE) == text
        assert attributes.pop(LLM_FINISH_REASON) == "stop"
        assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == usage["prompt_tokens"]
        assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == usage["completion_tokens"]
        assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == usage["total_tokens"]
        assert attributes == {}


def test_stream_without_usage_takes_tokens_from_invocation_metrics(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """gpt-oss streams have no ``usage`` unless ``stream_options.include_usage`` is set."""
    chunks = (
        {"choices": [{"delta": {"content": "pong"}, "finish_reason": None, "index": 0}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            "amazon-bedrock-invocationMetrics": {
                "inputTokenCount": 73,
                "outputTokenCount": 67,
                "invocationLatency": 1226,
                "firstByteLatency": 367,
            },
        },
    )
    events = [{"chunk": {"bytes": json.dumps(c).encode()}} for c in chunks]
    stream = MagicMock(spec=EventStream)
    stream.__iter__.return_value = iter(events)
    response = {"body": stream}
    span = tracer_provider.get_tracer(__name__).start_span(
        "bedrock.invoke_model_with_response_stream"
    )
    _InvokeModelWithResponseStream.handle_response(
        response,
        {
            "modelId": "openai.gpt-oss-120b-1:0",
            "body": json.dumps({"messages": [{"role": "user", "content": "say pong"}]}),
        },
        span,
    )
    for _ in response["body"]:
        pass

    (span_data,) = in_memory_span_exporter.get_finished_spans()
    assert span_data.status.is_ok
    attributes = dict(span_data.attributes or {})
    assert attributes.pop(OUTPUT_VALUE) == "pong"
    assert attributes.pop(LLM_FINISH_REASON) == "stop"
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 73
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 67
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 140


OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_FINISH_REASON = SpanAttributes.LLM_FINISH_REASON
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
