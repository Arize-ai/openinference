import io
from typing import (
    Generator,
)
from unittest.mock import MagicMock

import boto3
import pytest
from botocore.response import StreamingBody
from openinference.instrumentation.boto import BotoInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture(scope="module")
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> trace_api.TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


@pytest.fixture(scope="module")
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    BotoInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    BotoInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


def test_invoke_client(in_memory_span_exporter: InMemorySpanExporter):
    output = b'{"completion":" Hello!","stop_reason":"stop_sequence","stop":"\\n\\nHuman:"}'
    streaming_body = StreamingBody(io.BytesIO(output), len(output))
    mock_response = {
        "ResponseMetadata": {
            "RequestId": "xxxxxxxx-yyyy-zzzz-1234-abcdefghijklmno",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {
                "date": "Sun, 21 Jan 2024 20:00:00 GMT",
                "content-type": "application/json",
                "content-length": "74",
                "connection": "keep-alive",
                "x-amzn-requestid": "xxxxxxxx-yyyy-zzzz-1234-abcdefghijklmno",
                "x-amzn-bedrock-invocation-latency": "425",
                "x-amzn-bedrock-output-token-count": "6",
                "x-amzn-bedrock-input-token-count": "12",
            },
            "RetryAttempts": 0,
        },
        "contentType": "application/json",
        "body": streaming_body,
    }
    session = boto3.session.Session()
    client = session.client("bedrock-runtime", region_name="us-east-1")
    client._unwrapped_invoke_model = MagicMock(return_value=mock_response)
    client.invoke_model(
        modelId="anthropic.claude-v2",
        body=b'{"prompt": "Human: hello there? Assistant:", "max_tokens_to_sample": 1024}',
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans)  == 1
    span = spans[0]
    assert span.status.is_ok
    attributes = dict(span.attributes)
    assert attributes["llm.model_name"] == "anthropic.claude-v2"
    assert attributes["llm.prompts"] == "Human: hello there? Assistant:"
    assert attributes["message.content"] == " Hello!"
