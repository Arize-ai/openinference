import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, Tuple, cast
from unittest.mock import MagicMock

import aioboto3
import boto3
import pytest
from aioresponses import aioresponses
from botocore.eventstream import EventStream
from botocore.response import StreamingBody
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from openinference.instrumentation import REDACTED_VALUE, OITracer, TraceConfig
from openinference.instrumentation.bedrock import _model_invocation_wrapper
from openinference.instrumentation.bedrock._wrappers import _InvokeModelWithResponseStream
from openinference.instrumentation.bedrock.utils.anthropic._attributes import (
    get_llm_input_attributes,
)
from openinference.instrumentation.bedrock.utils.anthropic._messages import (
    _AnthropicMessagesCallback,
    _attributes_from_image_param,
    _attributes_from_request,
)

_CASSETTES_DIR = Path(__file__).resolve().parent / "cassettes"

_SECRET = "SECRET-SYSTEM"
_CLAUDE_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"


@pytest.mark.parametrize("stream", [False, True], ids=["invoke_model", "stream"])
@pytest.mark.parametrize(
    "system",
    [_SECRET, [{"type": "text", "text": _SECRET}]],
    ids=["string", "blocks"],
)
def test_hide_inputs_redacts_system_prompt(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    system: Any,
    stream: bool,
) -> None:
    _invoke_with_system(tracer_provider, TraceConfig(hide_inputs=True), system, stream)

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.is_ok
    attributes = dict(span.attributes or {})
    assert attributes.pop("input.value") == REDACTED_VALUE
    assert json.loads(str(attributes.pop("llm.invocation_parameters"))) == _invocation_parameters(
        stream
    )
    assert not [key for key in attributes if key.startswith("llm.input_messages.")]
    _pop_response_attributes(attributes, stream)
    assert not attributes


@pytest.mark.parametrize("stream", [False, True], ids=["invoke_model", "stream"])
@pytest.mark.parametrize(
    "system, texts",
    [
        (_SECRET, [_SECRET]),
        (
            [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
            ["first", "second"],
        ),
    ],
    ids=["string", "blocks"],
)
def test_system_prompt_is_first_input_message(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    system: Any,
    texts: list[str],
    stream: bool,
) -> None:
    _invoke_with_system(tracer_provider, TraceConfig(), system, stream)

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.is_ok
    attributes = dict(span.attributes or {})
    assert json.loads(str(attributes.pop("input.value")))["system"] == system
    assert attributes.pop("input.mime_type") == "application/json"
    assert json.loads(str(attributes.pop("llm.invocation_parameters"))) == _invocation_parameters(
        stream
    )
    assert attributes.pop("llm.input_messages.0.message.role") == "system"
    for i, text in enumerate(texts):
        prefix = f"llm.input_messages.0.message.contents.{i}.message_content"
        assert attributes.pop(f"{prefix}.text") == text
        if not stream:
            assert attributes.pop(f"{prefix}.type") == "text"
    assert attributes.pop("llm.input_messages.1.message.role") == "user"
    if stream:
        assert (
            attributes.pop("llm.input_messages.1.message.contents.0.message_content.text") == "hi"
        )
    else:
        assert attributes.pop("llm.input_messages.1.message.content") == "hi"
    _pop_response_attributes(attributes, stream)
    assert not attributes


def _invocation_parameters(stream: bool) -> Dict[str, Any]:
    # The streaming path also leaves anthropic_version out of the invocation parameters.
    if stream:
        return {"max_tokens": 10}
    return {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 10}


def _pop_response_attributes(attributes: Dict[str, Any], stream: bool) -> None:
    assert attributes.pop("openinference.span.kind") == "LLM"
    assert attributes.pop("llm.model_name") == _CLAUDE_MODEL_ID
    assert attributes.pop("llm.finish_reason") == "end_turn"
    assert attributes.pop("llm.token_count.prompt") == 5
    assert attributes.pop("llm.token_count.completion") == 1
    assert attributes.pop("llm.output_messages.0.message.role") == "assistant"
    if stream:
        output_text_key = "llm.output_messages.0.message.contents.0.message_content.text"
    else:
        output_text_key = "llm.output_messages.0.message.content"
        assert attributes.pop("llm.provider") == "aws"
        assert attributes.pop("llm.system") == "anthropic"
    assert attributes.pop(output_text_key) == "hello"
    assert attributes.pop("output.mime_type") == "application/json"
    assert json.loads(attributes.pop("output.value"))["content"][0]["text"] == "hello"


def _invoke_with_system(
    tracer_provider: TracerProvider,
    config: TraceConfig,
    system: Any,
    stream: bool,
) -> None:
    tracer = cast(Tracer, OITracer(tracer_provider.get_tracer(__name__), config=config))
    kwargs = {
        "modelId": _CLAUDE_MODEL_ID,
        "body": json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "system": system,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ),
    }
    if stream:
        chunks: Tuple[Dict[str, Any], ...] = (
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": _CLAUDE_MODEL_ID,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        )
        event_stream = MagicMock(spec=EventStream)
        event_stream.__iter__.return_value = iter(
            [{"chunk": {"bytes": json.dumps(c).encode()}} for c in chunks]
        )
        response = {"body": event_stream}
        _InvokeModelWithResponseStream(tracer)._sync_call(lambda **_: response, (), kwargs)
        for _ in response["body"]:
            pass
    else:
        payload = json.dumps(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": _CLAUDE_MODEL_ID,
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }
        ).encode()
        client = MagicMock()
        client._unwrapped_invoke_model.return_value = {
            "body": StreamingBody(io.BytesIO(payload), len(payload))
        }
        _model_invocation_wrapper(tracer)(client)(**kwargs)


def test_invoke_model_system_message_is_not_a_recorded_invocation_parameter() -> None:
    from openinference.instrumentation.bedrock.utils.anthropic._attributes import (
        _get_invocation_parameters,
    )

    kwargs = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": "SECRET-SYSTEM",
        "messages": [],
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "max_tokens": 10,
    }
    assert _get_invocation_parameters(kwargs) == {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10,
    }


def test_file_image_source_emits_no_attributes() -> None:
    image_block: Any = {
        "type": "image",
        "source": {"type": "file", "file_id": "file_123"},
    }

    assert list(_attributes_from_image_param(image_block, "prefix.")) == []


_TOOL = {
    "name": "lookup",
    "description": "look something up",
    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
}


@pytest.mark.parametrize(
    "system",
    [
        pytest.param("You are terse.", id="string"),
        pytest.param([{"type": "text", "text": "You are terse."}], id="text-blocks"),
    ],
)
def test_invoke_model_records_system_prompt_and_tools(system: Any) -> None:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 50,
        "system": system,
        "tools": [_TOOL],
        "messages": [{"role": "user", "content": "hello"}],
    }
    attributes = get_llm_input_attributes(body, "anthropic.claude-3-haiku-20240307-v1:0")

    assert attributes["llm.input_messages.0.message.role"] == "system"
    system_text = (
        attributes["llm.input_messages.0.message.content"]
        if isinstance(system, str)
        else attributes["llm.input_messages.0.message.contents.0.message_content.text"]
    )
    assert system_text == "You are terse."
    assert attributes["llm.input_messages.1.message.role"] == "user"
    assert attributes["llm.input_messages.1.message.content"] == "hello"
    assert json.loads(attributes["llm.tools.0.tool.json_schema"]) == _TOOL
    # the system prompt and tools are not duplicated in invocation parameters, where
    # hide_inputs / hide_input_messages would not mask them
    assert json.loads(attributes["llm.invocation_parameters"]) == {
        "max_tokens": 50,
    }


def test_streaming_request_records_system_text_blocks_under_input_messages() -> None:
    request = {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "body": {
            "system": [
                {"type": "text", "text": "You are terse."},
                {"type": "text", "text": "Answer in English."},
            ],
            "messages": [{"role": "user", "content": "hello"}],
        },
    }
    attributes = dict(_attributes_from_request(request))

    prefix = "llm.input_messages.0.message"
    assert attributes[f"{prefix}.role"] == "system"
    assert attributes[f"{prefix}.contents.0.message_content.text"] == "You are terse."
    assert attributes[f"{prefix}.contents.1.message_content.text"] == "Answer in English."
    assert not any(key.startswith("message.") for key in attributes)


@pytest.mark.parametrize(
    "system",
    [
        pytest.param("You are terse.", id="string"),
        pytest.param(
            [
                {"type": "text", "text": "You are terse."},
                {"type": "text", "text": "Answer in English."},
            ],
            id="multiple-blocks",
        ),
        pytest.param(None, id="absent"),
        pytest.param([], id="empty-blocks"),
    ],
)
def test_request_attributes_agree_across_paths(system: Any) -> None:
    body: Dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [_TOOL, {**_TOOL, "name": "lookup_second"}],
    }
    if system is not None:
        body["system"] = system
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    normal = get_llm_input_attributes(body, model_id)
    streaming = dict(_attributes_from_request({"modelId": model_id, "body": body}))

    for attributes in (normal, streaming):
        assert attributes["llm.model_name"] == "claude-3-haiku-20240307-v1:0"
        assert attributes["llm.provider"] == "aws"
        assert attributes["llm.system"] == "anthropic"
        assert json.loads(attributes["llm.invocation_parameters"]) == {"max_tokens": 50}
        for i, tool in enumerate(body["tools"]):
            assert json.loads(attributes[f"llm.tools.{i}.tool.json_schema"]) == tool
        assert attributes["llm.input_messages.0.message.role"] == "system" if system else "user"
        if system:
            assert attributes["llm.input_messages.1.message.role"] == "user"
            texts = [system] if isinstance(system, str) else [block["text"] for block in system]
            for i, expected in enumerate(texts):
                key = f"llm.input_messages.0.message.contents.{i}.message_content.text"
                if isinstance(system, str) and attributes is normal:
                    key = "llm.input_messages.0.message.content"
                assert attributes[key] == expected
        else:
            assert not any(k.startswith("llm.input_messages.1.") for k in attributes)
        assert not any(k.startswith("message.contents.") for k in attributes)


@pytest.mark.parametrize("has_response", [True, False])
def test_stream_final_model_name(
    has_response: bool,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    span = tracer_provider.get_tracer(__name__).start_span("stream")
    callback = _AnthropicMessagesCallback(
        span,
        {"modelId": "anthropic.claude-3-haiku-20240307-v1:0", "body": {"messages": []}},
    )
    if has_response:
        event = {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-haiku",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 7, "output_tokens": 0},
            },
        }
        callback({"chunk": {"bytes": json.dumps(event).encode()}})
    callback(StopIteration())
    (finished,) = in_memory_span_exporter.get_finished_spans()
    assert finished.attributes is not None
    assert finished.attributes["llm.model_name"] == (
        "claude-3-haiku" if has_response else "claude-3-haiku-20240307-v1:0"
    )
    assert finished.attributes["llm.provider"] == "aws"
    assert finished.attributes["llm.system"] == "anthropic"


def _assert_invoke_model_span_attributes(
    attributes: Dict[str, Any],
    prompt: Dict[str, Any],
) -> None:
    assert json.loads(attributes.pop("input.value")) == prompt
    assert attributes.pop("llm.input_messages.0.message.content") == "Hello there."
    assert attributes.pop("llm.input_messages.0.message.role") == "user"
    assert (
        attributes.pop("llm.input_messages.1.message.content")
        == "Hi, I'm Claude. How can I help you?"
    )
    assert attributes.pop("llm.input_messages.1.message.role") == "assistant"
    assert (
        attributes.pop("llm.input_messages.2.message.content")
        == "Can you explain LLMs in plain English?"
    )
    assert attributes.pop("llm.input_messages.2.message.role") == "user"
    assert attributes.pop("llm.invocation_parameters") == '{"max_tokens": 1000, "temperature": 0.7}'
    assert attributes.pop("llm.model_name") == "claude-3-haiku-20240307"
    assert attributes.pop("llm.finish_reason") == "end_turn"
    assert attributes.pop("llm.output_messages.0.message.role") == "assistant"
    assert "LLMs are a type of artificial intelligence that" in attributes.pop(
        "llm.output_messages.0.message.content"
    )
    assert attributes.pop("llm.provider") == "aws"
    assert attributes.pop("llm.system") == "anthropic"
    assert attributes.pop("llm.token_count.completion") == 246
    assert attributes.pop("llm.token_count.prompt") == 38
    assert attributes.pop("openinference.span.kind") == "LLM"
    assert attributes.pop("output.mime_type") == "application/json"
    assert attributes.pop("input.mime_type") == "application/json"
    output_value = json.loads(attributes.pop("output.value"))
    assert output_value["role"] == "assistant"
    assert output_value["model"] == "claude-3-haiku-20240307"
    assert output_value["usage"]["input_tokens"] == 38
    assert output_value["usage"]["output_tokens"] == 246
    assert not attributes, f"Unexpected leftover attributes: {attributes}"


def _assert_invoke_model_with_image_span_attributes(attributes: Dict[str, Any]) -> None:
    assert attributes.pop("input.mime_type") == "application/json"
    input_value = json.loads(attributes.pop("input.value"))
    assert input_value["messages"][0]["role"] == "user"
    assert input_value["messages"][0]["content"][0]["type"] == "text"
    assert input_value["messages"][0]["content"][0]["text"] == "What's in this image?"
    assert input_value["messages"][0]["content"][1]["type"] == "image"
    assert input_value["messages"][0]["content"][1]["source"]["type"] == "base64"
    assert input_value["messages"][0]["content"][1]["source"]["media_type"] == "image/webp"
    assert input_value["messages"][0]["content"][1]["source"]["data"].startswith("R0lGOD")
    assert attributes.pop("llm.input_messages.0.message.contents.1.message_content.type") == "image"
    assert attributes.pop("llm.input_messages.0.message.role") == "user"
    assert attributes.pop("llm.invocation_parameters") == '{"max_tokens": 1000, "temperature": 0.7}'
    assert attributes.pop("llm.model_name") == "claude-3-haiku-20240307"
    assert attributes.pop("llm.finish_reason") == "end_turn"
    assert attributes.pop("llm.output_messages.0.message.role") == "assistant"
    assert "Homer Simpson" in attributes.pop("llm.output_messages.0.message.content")
    assert attributes.pop("llm.provider") == "aws"
    assert attributes.pop("llm.system") == "anthropic"
    assert attributes.pop("llm.token_count.completion") == 76
    assert attributes.pop("llm.token_count.prompt") == 480
    assert attributes.pop("openinference.span.kind") == "LLM"
    assert attributes.pop("output.mime_type") == "application/json"
    output = json.loads(attributes.pop("output.value"))
    assert output["role"] == "assistant"
    assert output["model"] == "claude-3-haiku-20240307"
    assert output["usage"]["input_tokens"] == 480
    assert output["usage"]["output_tokens"] == 76
    assert output["content"][0]["type"] == "text"
    assert "Homer Simpson" in output["content"][0]["text"]
    prefix = "llm.input_messages.0.message.contents"
    assert attributes.pop(f"{prefix}.0.message_content.type") == "text"
    assert attributes.pop(f"{prefix}.0.message_content.text") == "What's in this image?"
    assert attributes.pop(f"{prefix}.1.message_content.image.image.url").startswith(
        "data:image/webp;base64,R0lGOD"
    )
    assert not attributes, f"Unexpected leftover attributes: {attributes}"


class TestClaudeInvokeModelMessageApi:
    @pytest.mark.vcr(
        before_record_request=lambda _: _.headers.clear() or _,
    )
    def test_invoke_model(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        prompt = {
            "messages": [
                {"role": "user", "content": "Hello there."},
                {"role": "assistant", "content": "Hi, I'm Claude. How can I help you?"},
                {"role": "user", "content": "Can you explain LLMs in plain English?"},
            ],
            "max_tokens": 1000,
            "temperature": 0.7,
            "anthropic_version": "bedrock-2023-05-31",
        }
        session = boto3.session.Session()
        client = session.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="123",
            aws_secret_access_key="321",
        )
        response = client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0", body=json.dumps(prompt)
        )
        json.loads(response.get("body").read())
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.is_ok
        _assert_invoke_model_span_attributes(dict(spans[0].attributes or {}), prompt)

    @pytest.mark.order(after="test_invoke_model")
    async def test_async_invoke_model(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        read_aio_cassette: Any,
    ) -> None:
        """Async version of test_invoke_model; same cassette and assertions."""
        prompt = {
            "messages": [
                {"role": "user", "content": "Hello there."},
                {"role": "assistant", "content": "Hi, I'm Claude. How can I help you?"},
                {"role": "user", "content": "Can you explain LLMs in plain English?"},
            ],
            "max_tokens": 1000,
            "temperature": 0.7,
            "anthropic_version": "bedrock-2023-05-31",
        }
        with aioresponses() as m:
            read_aio_cassette(
                str(_CASSETTES_DIR / "TestClaudeInvokeModelMessageApi.test_invoke_model.yaml"),
                m,
            )
            session = aioboto3.session.Session()
            async with session.client(
                "bedrock-runtime",
                region_name="us-east-1",
                aws_access_key_id="123",
                aws_secret_access_key="321",
            ) as client:
                response = await client.invoke_model(
                    modelId="anthropic.claude-3-haiku-20240307-v1:0",
                    body=json.dumps(prompt),
                )
                body = response.get("body")
                data = await body.read()
                json.loads(data.decode())
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.is_ok
        _assert_invoke_model_span_attributes(dict(spans[0].attributes or {}), prompt)

    @pytest.mark.vcr(
        before_record_request=lambda _: _.headers.clear() or _,
    )
    def test_invoke_model_with_image(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        image_bytes_and_format: Tuple[bytes, str],
    ) -> None:
        input_text = "What's in this image?"
        img_bytes, img_format = image_bytes_and_format
        message = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_text,
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{img_format}",
                                "data": base64.b64encode(img_bytes).decode("utf-8"),
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7,
            "anthropic_version": "bedrock-2023-05-31",
        }
        session = boto3.session.Session()
        client = session.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="123",
            aws_secret_access_key="321",
        )
        response = client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0", body=json.dumps(message)
        )
        json.loads(response.get("body").read())
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.is_ok
        _assert_invoke_model_with_image_span_attributes(dict(spans[0].attributes or {}))

    @pytest.mark.order(after="test_invoke_model_with_image")
    async def test_async_invoke_model_with_image(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        read_aio_cassette: Any,
        image_bytes_and_format: Tuple[bytes, str],
    ) -> None:
        """Async version of test_invoke_model_with_image; same cassette and assertions."""
        input_text = "What's in this image?"
        img_bytes, img_format = image_bytes_and_format
        message = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_text},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{img_format}",
                                "data": base64.b64encode(img_bytes).decode("utf-8"),
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7,
            "anthropic_version": "bedrock-2023-05-31",
        }
        with aioresponses() as m:
            read_aio_cassette(
                str(
                    _CASSETTES_DIR
                    / "TestClaudeInvokeModelMessageApi.test_invoke_model_with_image.yaml"
                ),
                m,
            )
            session = aioboto3.session.Session()
            async with session.client(
                "bedrock-runtime",
                region_name="us-east-1",
                aws_access_key_id="123",
                aws_secret_access_key="321",
            ) as client:
                response = await client.invoke_model(
                    modelId="anthropic.claude-3-haiku-20240307-v1:0",
                    body=json.dumps(message),
                )
                body = response.get("body")
                data = await body.read()
                json.loads(data.decode())
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.is_ok
        _assert_invoke_model_with_image_span_attributes(dict(spans[0].attributes or {}))
