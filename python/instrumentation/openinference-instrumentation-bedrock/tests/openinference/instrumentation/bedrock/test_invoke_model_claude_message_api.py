import base64
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import aioboto3
import boto3
import pytest
from aioresponses import aioresponses
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_CASSETTES_DIR = Path(__file__).resolve().parent / "cassettes"


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
    assert (
        attributes.pop("llm.invocation_parameters")
        == '{"max_tokens": 1000, "temperature": 0.7, "anthropic_version": '
        '"bedrock-2023-05-31"}'
    )
    assert attributes.pop("llm.model_name") == "claude-3-haiku-20240307"
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
    assert (
        attributes.pop("llm.invocation_parameters") == '{"max_tokens": 1000, "temperature": 0.7, '
        '"anthropic_version": "bedrock-2023-05-31"}'
    )
    assert attributes.pop("llm.model_name") == "claude-3-haiku-20240307"
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
