import base64
import json
import os
from typing import Any, Dict

import boto3
import pytest
import requests
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def sanitize_format(fmt: str) -> str:
    return "jpeg" if fmt == "jpg" else fmt


def download_img(url: str) -> Any:
    img_format = sanitize_format(os.path.splitext(url)[-1].lstrip("."))
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError(f"Error: Could not retrieve image from URL: {url}")
    return resp.content, img_format


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
        span = spans[0]
        assert span.status.is_ok
        attributes: Dict[str, Any] = dict(span.attributes or {})
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
        )  # partial match is safer for long text

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

        # Finally, make sure no unexpected attributes remain
        assert not attributes, f"Unexpected leftover attributes: {attributes}"

    @pytest.mark.vcr(
        before_record_request=lambda _: _.headers.clear() or _,
    )
    def test_invoke_model_with_image(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        input_text = "What's in this image?"

        img_url = (
            "https://a1cf74336522e87f135f-2f21ace9a6cf0052456644b80fa06d4f.ssl.cf2.rackcdn.com"
            "/images/characters/large/800/Homer-Simpson.The-Simpsons.webp"
        )
        img_bytes, img_format = download_img(img_url)

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
        span = spans[0]
        assert span.status.is_ok
        attributes: dict[str, Any] = dict(span.attributes or {})
        assert attributes.pop("input.mime_type") == "application/json"

        input_value = json.loads(attributes.pop("input.value"))
        assert input_value["messages"][0]["role"] == "user"
        assert input_value["messages"][0]["content"][0]["type"] == "text"
        assert input_value["messages"][0]["content"][0]["text"] == "What's in this image?"
        assert input_value["messages"][0]["content"][1]["type"] == "image"
        assert input_value["messages"][0]["content"][1]["source"]["type"] == "base64"
        assert input_value["messages"][0]["content"][1]["source"]["media_type"] == "image/webp"
        assert input_value["messages"][0]["content"][1]["source"]["data"].startswith("UklGRt4+")

        assert (
            attributes.pop("llm.input_messages.0.message.contents.1.message_content.type")
            == "image"
        )
        assert attributes.pop("llm.input_messages.0.message.role") == "user"
        assert (
            attributes.pop("llm.invocation_parameters")
            == '{"max_tokens": 1000, "temperature": 0.7, '
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
            "data:image/webp;base64,UklGRt4+"
        )
        # Make sure everything is consumed
        assert not attributes, f"Unexpected leftover attributes: {attributes}"
