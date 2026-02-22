import asyncio
import base64
import json
import os

import aioboto3
import requests
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.bedrock import BedrockInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

BedrockInstrumentor().instrument(tracer_provider=tracer_provider)


async def claude3_invoke_model():
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
    session = aioboto3.session.Session(
        region_name="us-east-1",
    )
    async with session.client("bedrock-runtime") as client:
        response = await client.invoke_model_with_response_stream(
            modelId="us.anthropic.claude-sonnet-4-6", body=json.dumps(prompt)
        )
        async for chunk in response["body"]:
            print(chunk)


def sanitize_format(fmt: str) -> str:
    return "jpeg" if fmt == "jpg" else fmt


def download_img(url: str):
    img_format = sanitize_format(os.path.splitext(url)[-1].lstrip("."))
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError(f"Error: Could not retrieve image from URL: {url}")
    return resp.content, img_format


async def invoke_image_call():
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
    session = aioboto3.session.Session(
        region_name="us-east-1",
    )
    async with session.client("bedrock-runtime") as client:
        response = await client.invoke_model_with_response_stream(
            modelId="anthropic.claude-3-haiku-20240307-v1:0", body=json.dumps(message)
        )
        async for chunk in response["body"]:
            print(chunk)


if __name__ == "__main__":
    # asyncio.run(invoke_image_call())
    asyncio.run(claude3_invoke_model())
