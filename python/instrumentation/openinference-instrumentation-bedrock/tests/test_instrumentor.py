import io
import json
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock

import boto3
import pytest
from botocore.response import StreamingBody
from openinference.instrumentation import using_attributes
from openinference.instrumentation.bedrock import BedrockInstrumentor
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture()
def session_id() -> str:
    return "my-test-session-id"


@pytest.fixture()
def user_id() -> str:
    return "my-test-user-id"


@pytest.fixture()
def metadata() -> Dict[str, Any]:
    return {
        "test-int": 1,
        "test-str": "string",
        "test-list": [1, 2, 3],
        "test-dict": {
            "key-1": "val-1",
            "key-2": "val-2",
        },
    }


@pytest.fixture()
def tags() -> List[str]:
    return ["tag-1", "tag-2"]


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
    BedrockInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    BedrockInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_invoke_client(
    use_context_attributes: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
) -> None:
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
    # instead of mocking the HTTP response, we mock the boto client method directly to avoid
    # complexities with mocking auth
    client._unwrapped_invoke_model = MagicMock(return_value=mock_response)
    body = {"prompt": "Human: hello there? Assistant:", "max_tokens_to_sample": 1024}
    model_name = "anthropic.claude-v2"
    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
        ):
            client.invoke_model(
                modelId=model_name,
                body=json.dumps(body),
            )
    else:
        client.invoke_model(
            modelId=model_name,
            body=json.dumps(body),
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.is_ok
    attributes = dict(span.attributes or dict())
    print(attributes)
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert attributes.pop(INPUT_VALUE) == body["prompt"]
    assert attributes.pop(OUTPUT_VALUE) == " Hello!"
    assert attributes.pop(LLM_MODEL_NAME) == model_name
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 12
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 6
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 18
    assert isinstance(invocation_parameters_str := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(invocation_parameters_str) == {
        "max_tokens_to_sample": 1024,
    }
    if use_context_attributes:
        _check_context_attributes(attributes, session_id, user_id, metadata, tags)
    assert attributes == {}


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_invoke_client_with_missing_tokens(
    use_context_attributes: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
) -> None:
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
            },
            "RetryAttempts": 0,
        },
        "contentType": "application/json",
        "body": streaming_body,
    }
    session = boto3.session.Session()
    client = session.client("bedrock-runtime", region_name="us-east-1")

    # instead of mocking the HTTP response, we mock the boto client method directly to avoid
    # complexities with mocking auth
    client._unwrapped_invoke_model = MagicMock(return_value=mock_response)
    body = {"prompt": "Human: hello there? Assistant:", "max_tokens_to_sample": 1024}
    model_name = "anthropic.claude-v2"
    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
        ):
            client.invoke_model(
                modelId=model_name,
                body=json.dumps(body),
            )
    else:
        client.invoke_model(
            modelId=model_name,
            body=json.dumps(body),
        )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.is_ok
    attributes = dict(span.attributes or dict())
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert attributes.pop(INPUT_VALUE) == body["prompt"]
    assert attributes.pop(OUTPUT_VALUE) == " Hello!"
    assert attributes.pop(LLM_MODEL_NAME) == model_name
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 6
    assert isinstance(invocation_parameters_str := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(invocation_parameters_str) == {
        "max_tokens_to_sample": 1024,
    }
    if use_context_attributes:
        _check_context_attributes(attributes, session_id, user_id, metadata, tags)
    assert attributes == {}


def _check_context_attributes(
    attributes: Dict[str, Any],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
) -> None:
    assert attributes.pop(SESSION_ID, None) == session_id
    assert attributes.pop(USER_ID, None) == user_id
    attr_metadata = attributes.pop(METADATA, None)
    assert attr_metadata is not None
    assert isinstance(attr_metadata, str)  # must be json string
    metadata_dict = json.loads(attr_metadata)
    assert metadata_dict == metadata
    attr_tags = attributes.pop(TAG_TAGS, None)
    assert attr_tags is not None
    assert len(attr_tags) == len(tags)
    assert list(attr_tags) == tags


OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
# DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
# DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
# DOCUMENT_METADATA = DocumentAttributes.DOCUMENT_METADATA
# EMBEDDING_EMBEDDINGS = SpanAttributes.EMBEDDING_EMBEDDINGS
# EMBEDDING_MODEL_NAME = SpanAttributes.EMBEDDING_MODEL_NAME
# EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
# EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
# LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
# LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
# LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
# LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
# LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
# MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
# MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
# MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
# MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
# MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
METADATA = SpanAttributes.METADATA
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
# RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
# TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
# TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
# LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
# LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES
#
# CHAIN = OpenInferenceSpanKindValues.CHAIN
# LLM = OpenInferenceSpanKindValues.LLM
# RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER
#
JSON = OpenInferenceMimeTypeValues.JSON
SESSION_ID = SpanAttributes.SESSION_ID
USER_ID = SpanAttributes.USER_ID
TAG_TAGS = SpanAttributes.TAG_TAGS
