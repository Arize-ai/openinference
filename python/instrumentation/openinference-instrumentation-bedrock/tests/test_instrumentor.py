import base64
import io
import json
from typing import Any, Dict, Generator, List, Tuple
from unittest.mock import MagicMock

import boto3
import pytest
from botocore.response import StreamingBody
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation import OITracer, using_attributes
from openinference.instrumentation.bedrock import (
    _MINIMUM_CONVERSE_BOTOCORE_VERSION,
    BedrockInstrumentor,
)
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


@pytest.fixture()
def image_bytes_and_format() -> Tuple[bytes, str]:
    return (
        b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;",  # noqa: E501
        "webp",
    )


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


@pytest.fixture
def prompt_template() -> str:
    return (
        "This is a test prompt template with int {var_int}, "
        "string {var_string}, and list {var_list}"
    )


@pytest.fixture
def prompt_template_version() -> str:
    return "v1.0"


@pytest.fixture
def prompt_template_variables() -> Dict[str, Any]:
    return {
        "var_int": 1,
        "var_str": "2",
        "var_list": [1, 2, 3],
    }


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


class TestInstrumentor:
    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        (instrumentor_entrypoint,) = entry_points(
            group="opentelemetry_instrumentor", name="bedrock"
        )
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, BedrockInstrumentor)

    # Ensure we're using the common OITracer from common openinference-instrumentation pkg
    def test_oitracer(self) -> None:
        assert isinstance(BedrockInstrumentor()._tracer, OITracer)


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_invoke_client(
    use_context_attributes: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
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
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
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
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 12
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 6
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 18
    assert isinstance(invocation_parameters_str := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(invocation_parameters_str) == {
        "max_tokens_to_sample": 1024,
    }
    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_invoke_client_with_missing_tokens(
    use_context_attributes: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
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
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
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
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_converse(
    use_context_attributes: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    if version := boto3.__version__ < _MINIMUM_CONVERSE_BOTOCORE_VERSION:
        pytest.xfail(
            f"Botocore {version} does not support the Converse API. "
            f"Converse API introduced in {_MINIMUM_CONVERSE_BOTOCORE_VERSION}"
        )
    system = [{"text": "return a short response"}]
    inference_config = {"maxTokens": 1024, "temperature": 0.0}
    output = {
        "message": {
            "role": "assistant",
            "content": [{"text": "Hi there! How can I assist you today?"}],
        }
    }
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
        "output": output,
        "usage": {"inputTokens": 12, "outputTokens": 6, "totalTokens": 18},
    }
    session = boto3.session.Session()
    client = session.client("bedrock-runtime", region_name="us-east-1")
    # instead of mocking the HTTP response, we mock the boto client method directly to avoid
    # complexities with mocking auth
    client._unwrapped_converse = MagicMock(return_value=mock_response)
    message = {"role": "user", "content": [{"text": "hello there?"}]}
    model_name = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            client.converse(
                modelId=model_name,
                system=system,
                messages=[message],
                inferenceConfig=inference_config,
            )
    else:
        client.converse(
            modelId=model_name, system=system, messages=[message], inferenceConfig=inference_config
        )

    llm_input_messages_truths = [
        {"role": "system", "content": system[0]["text"]},
        {"role": message.get("role"), "content": message.get("content", [{}])[0].get("text")},  # type: ignore
    ]

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    _run_converse_checks(
        use_context_attributes,
        session_id,
        user_id,
        metadata,
        tags,
        prompt_template,
        prompt_template_version,
        prompt_template_variables,
        span=spans[0],
        llm_input_messages_truth=llm_input_messages_truths,
        input=message["content"][0]["text"],  # type: ignore
        output=output,
        model_name=model_name,
        token_counts=mock_response["usage"],  # type: ignore
        invocation_parameters=inference_config,
    )


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_converse_multiple(
    use_context_attributes: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    if version := boto3.__version__ < _MINIMUM_CONVERSE_BOTOCORE_VERSION:
        pytest.xfail(
            f"Botocore {version} does not support the Converse API. "
            f"Converse API introduced in {_MINIMUM_CONVERSE_BOTOCORE_VERSION}"
        )
    first_output = {
        "message": {
            "role": "assistant",
            "content": [{"text": "Hi there! How can I assist you today?"}],
        }
    }
    first_mock_response = {
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
        "output": first_output,
        "usage": {"inputTokens": 12, "outputTokens": 6, "totalTokens": 18},
    }

    session = boto3.session.Session()
    client = session.client("bedrock-runtime", region_name="us-east-1")
    # instead of mocking the HTTP response, we mock the boto client method directly to avoid
    # complexities with mocking auth
    client._unwrapped_converse = MagicMock(return_value=first_mock_response)

    system = [{"text": "return a short response."}, {"text": "clarify your responses."}]
    inference_config = {"maxTokens": 1024, "temperature": 0.0}
    first_msg = {"role": "user", "content": [{"text": "hello there?"}]}
    second_msg = {"role": "user", "content": [{"text": "how are you?"}]}
    model_name = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    messages = [first_msg]
    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            response = client.converse(
                modelId=model_name,
                system=system,
                messages=messages,
                inferenceConfig=inference_config,
            )
    else:
        response = client.converse(
            modelId=model_name, system=system, messages=messages, inferenceConfig=inference_config
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    system_prompt1, system_prompt2 = system[0].get("text"), system[1].get("text")
    llm_input_messages_truths = [
        {"role": "system", "content": f"{system_prompt1} {system_prompt2}"},
        {"role": first_msg["role"], "content": first_msg["content"][0]["text"]},  # type: ignore
    ]
    _run_converse_checks(
        use_context_attributes,
        session_id,
        user_id,
        metadata,
        tags,
        prompt_template,
        prompt_template_version,
        prompt_template_variables,
        span=spans[0],
        llm_input_messages_truth=llm_input_messages_truths,
        input=first_msg["content"][0]["text"],  # type: ignore
        output=first_output,
        model_name=model_name,
        token_counts=first_mock_response["usage"],  # type: ignore
        invocation_parameters=inference_config,
    )

    second_output = {
        "message": {
            "role": "assistant",
            "content": [
                {"text": "I'm functioning well, thank you for asking. How are you doing today?"},
            ],
        }
    }
    second_mock_response = {
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
        "output": second_output,
        "usage": {"inputTokens": 12, "outputTokens": 6, "totalTokens": 18},
    }
    client._unwrapped_converse = MagicMock(return_value=second_mock_response)
    messages.extend([response["output"]["message"], second_msg])
    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            client.converse(
                modelId=model_name,
                system=system,
                messages=messages,
                inferenceConfig=inference_config,
            )
    else:
        client.converse(
            modelId=model_name, system=system, messages=messages, inferenceConfig=inference_config
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    llm_input_messages_truths.extend(
        [
            {
                "role": first_output["message"]["role"],
                "content": first_output["message"]["content"][0]["text"],  # type: ignore
            },
            {"role": second_msg["role"], "content": second_msg["content"][0]["text"]},  # type: ignore
        ]
    )
    _run_converse_checks(
        use_context_attributes,
        session_id,
        user_id,
        metadata,
        tags,
        prompt_template,
        prompt_template_version,
        prompt_template_variables,
        span=spans[1],
        llm_input_messages_truth=llm_input_messages_truths,
        input=second_msg["content"][0]["text"],  # type: ignore
        output=second_output,
        model_name=model_name,
        token_counts=second_mock_response["usage"],  # type: ignore
        invocation_parameters=inference_config,
    )


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_converse_with_missing_tokens(
    use_context_attributes: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    if version := boto3.__version__ < _MINIMUM_CONVERSE_BOTOCORE_VERSION:
        pytest.xfail(
            f"Botocore {version} does not support the Converse API. "
            f"Converse API introduced in {_MINIMUM_CONVERSE_BOTOCORE_VERSION}"
        )
    system = [{"text": "return a short response"}]
    inference_config = {"maxTokens": 1024, "temperature": 0.0}
    output = {
        "message": {
            "role": "assistant",
            "content": [{"text": "Hi there! How can I assist you today?"}],
        }
    }
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
        "output": output,
        "usage": {"outputTokens": 6},
    }
    session = boto3.session.Session()
    client = session.client("bedrock-runtime", region_name="us-east-1")
    # instead of mocking the HTTP response, we mock the boto client method directly to avoid
    # complexities with mocking auth
    client._unwrapped_converse = MagicMock(return_value=mock_response)
    message = {"role": "user", "content": [{"text": "hello there?"}]}
    model_name = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            client.converse(
                modelId=model_name,
                system=system,
                messages=[message],
                inferenceConfig=inference_config,
            )
    else:
        client.converse(
            modelId=model_name, system=system, messages=[message], inferenceConfig=inference_config
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    llm_input_messages_truths = [
        {"role": "system", "content": system[0]["text"]},
        {"role": message["role"], "content": message["content"][0]["text"]},  # type: ignore
    ]
    _run_converse_checks(
        use_context_attributes,
        session_id,
        user_id,
        metadata,
        tags,
        prompt_template,
        prompt_template_version,
        prompt_template_variables,
        span=spans[0],
        llm_input_messages_truth=llm_input_messages_truths,
        input=message["content"][0]["text"],  # type: ignore
        output=output,
        model_name=model_name,
        token_counts=mock_response["usage"],  # type: ignore
        invocation_parameters=inference_config,
    )


@pytest.mark.parametrize("use_context_attributes", [False, True])
@pytest.mark.parametrize(
    "model_id",
    [
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "mistral.mistral-large-2402-v1:0",
        "mistral.mistral-small-2402-v1:0",
        "meta.llama3-8b-instruct-v1:0",
        "meta.llama3-70b-instruct-v1:0",
    ],
)
def test_converse_multiple_models(
    use_context_attributes: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
    model_id: str,
) -> None:
    if version := boto3.__version__ < _MINIMUM_CONVERSE_BOTOCORE_VERSION:
        pytest.xfail(
            f"Botocore {version} does not support the Converse API. "
            f"Converse API introduced in {_MINIMUM_CONVERSE_BOTOCORE_VERSION}"
        )
    inference_config = {"maxTokens": 1024, "temperature": 0.0}
    output = {
        "message": {
            "role": "assistant",
            "content": [{"text": "Hi there! How can I assist you today?"}],
        }
    }
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
        "output": output,
        "usage": {"inputTokens": 12, "outputTokens": 6, "totalTokens": 18},
    }
    session = boto3.session.Session()
    client = session.client("bedrock-runtime", region_name="us-east-1")
    # instead of mocking the HTTP response, we mock the boto client method directly to avoid
    # complexities with mocking auth
    client._unwrapped_converse = MagicMock(return_value=mock_response)
    message = {"role": "user", "content": [{"text": "hello there?"}]}
    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            client.converse(
                modelId=model_id,
                messages=[message],
                inferenceConfig=inference_config,
            )
    else:
        client.converse(modelId=model_id, messages=[message], inferenceConfig=inference_config)

    llm_input_messages_truths = [
        {"role": message["role"], "content": message["content"][0]["text"]},  # type: ignore
    ]

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    _run_converse_checks(
        use_context_attributes,
        session_id,
        user_id,
        metadata,
        tags,
        prompt_template,
        prompt_template_version,
        prompt_template_variables,
        span=spans[0],
        llm_input_messages_truth=llm_input_messages_truths,
        input=message["content"][0]["text"],  # type: ignore
        output=output,
        model_name=model_id,
        token_counts=mock_response["usage"],  # type: ignore
        invocation_parameters=inference_config,
    )


@pytest.mark.parametrize("use_context_attributes", [False])  # , True])
def test_converse_multimodal(
    use_context_attributes: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    image_bytes_and_format: Tuple[bytes, str],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    if version := boto3.__version__ < _MINIMUM_CONVERSE_BOTOCORE_VERSION:
        pytest.xfail(
            f"Botocore {version} does not support the Converse API. "
            f"Converse API introduced in {_MINIMUM_CONVERSE_BOTOCORE_VERSION}"
        )
    system = [{"text": "return a short response"}]
    inference_config = {"maxTokens": 1024, "temperature": 0.0}
    output = {
        "message": {
            "role": "assistant",
            "content": [{"text": "Hi there! How can I assist you today?"}],
        }
    }
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
        "output": output,
        "usage": {"inputTokens": 12, "outputTokens": 6, "totalTokens": 18},
    }
    session = boto3.session.Session()
    client = session.client("bedrock-runtime", region_name="us-east-1")
    # instead of mocking the HTTP response, we mock the boto client method directly to avoid
    # complexities with mocking auth
    client._unwrapped_converse = MagicMock(return_value=mock_response)
    model_name = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    input_text = "What's in this image?"

    img_bytes, format = image_bytes_and_format
    message = {
        "role": "user",
        "content": [
            {
                "text": input_text,
            },
            {
                "image": {
                    "format": format,
                    "source": {
                        "bytes": img_bytes,
                    },
                }
            },
        ],
    }

    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            client.converse(
                modelId=model_name,
                system=system,
                messages=[message],
                inferenceConfig=inference_config,
            )
    else:
        client.converse(
            modelId=model_name,
            system=system,
            messages=[message],
            inferenceConfig=inference_config,
        )

    llm_input_messages_truths = [
        {"role": "system", "content": system[0]["text"]},
        {"role": message.get("role"), "content": message.get("content")},
    ]

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    _run_converse_checks(
        use_context_attributes,
        session_id,
        user_id,
        metadata,
        tags,
        prompt_template,
        prompt_template_version,
        prompt_template_variables,
        span=spans[0],
        llm_input_messages_truth=llm_input_messages_truths,  # type:ignore
        input=message["content"][0]["text"],  # type: ignore
        output=output,
        model_name=model_name,
        token_counts=mock_response["usage"],  # type: ignore
        invocation_parameters=inference_config,
    )


def _check_context_attributes(
    attributes: Dict[str, Any],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
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
    assert attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE, None) == prompt_template
    assert (
        attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION, None) == prompt_template_version
    )
    assert attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES, None) == json.dumps(
        prompt_template_variables
    )


def _run_converse_checks(
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
    span: trace_sdk.ReadableSpan,
    llm_input_messages_truth: List[Dict[str, str]],
    input: str,
    output: Dict[str, Any],
    model_name: str,
    token_counts: Dict[Any, Any],
    invocation_parameters: Dict[str, Any],
) -> None:
    assert span.status.is_ok
    attributes = dict(span.attributes or dict())
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert attributes.pop(LLM_MODEL_NAME) == model_name

    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT, None) == token_counts.get("inputTokens")
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION, None) == token_counts.get("outputTokens")
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL, None) == token_counts.get("totalTokens")

    assert isinstance(invocation_parameters_str := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(invocation_parameters_str) == invocation_parameters

    assert attributes.pop(OUTPUT_VALUE) == output["message"]["content"][0]["text"]
    assert attributes.pop("llm.output_messages.0.message.role") == output["message"]["role"]
    assert (
        attributes.pop("llm.output_messages.0.message.content")
        == output["message"]["content"][0]["text"]
    )

    assert attributes.pop(INPUT_VALUE) == input
    for msg_idx, msg in enumerate(llm_input_messages_truth):
        role_key = f"llm.input_messages.{msg_idx}.message.role"
        content_key = f"llm.input_messages.{msg_idx}.message.contents"
        assert attributes.pop(role_key) == msg["role"], f"Role mismatch for message {msg_idx}."
        content = msg["content"]
        if isinstance(content, str):
            content = [{"text": content}]  # type:ignore
        for content_idx, content_item in enumerate(content):
            if content_item.get("image"):  # type:ignore
                expected_type = "image"
                base64_img = base64.b64encode(content_item["image"]["source"]["bytes"]).decode(  # type:ignore
                    "utf-8"
                )
                expected_value = f"data:image/jpeg;base64,{base64_img}"
                value_key = f"{content_key}.{content_idx}.message_content.image.image.url"
            else:
                expected_type = "text"
                expected_value = content_item["text"]  # type:ignore
                value_key = f"{content_key}.{content_idx}.message_content.text"
            type_key = f"{content_key}.{content_idx}.message_content.type"
            assert attributes.pop(type_key) == expected_type
            assert attributes.pop(value_key) == expected_value

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}


OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
METADATA = SpanAttributes.METADATA
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
JSON = OpenInferenceMimeTypeValues.JSON
SESSION_ID = SpanAttributes.SESSION_ID
USER_ID = SpanAttributes.USER_ID
TAG_TAGS = SpanAttributes.TAG_TAGS
