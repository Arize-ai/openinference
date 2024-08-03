import os
from typing import Tuple

import boto3
import requests
from openinference.instrumentation.bedrock import BedrockInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace_api.set_tracer_provider(tracer_provider=tracer_provider)

BedrockInstrumentor().instrument()

session = boto3.session.Session()
client = session.client("bedrock-runtime", "us-east-1")


def multimodal_example():
    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    input_text = "What's in this image?"

    img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    img_url = "https://a1cf74336522e87f135f-2f21ace9a6cf0052456644b80fa06d4f.ssl.cf2.rackcdn.com/images/characters/large/800/Homer-Simpson.The-Simpsons.webp"
    img_bytes, format = download_img(img_url)

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

    response = client.converse(
        modelId=model_id,
        messages=[message],
    )

    out = response["output"]["message"]
    print(out.get("content")[-1].get("text"))


def download_img(url: str) -> Tuple[bytes, str]:
    format = sanitize_format(os.path.splitext(url)[-1].lstrip("."))
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError(f"Error: Could not retrieve image from URL: {url}")
    return resp.content, format


def sanitize_format(fmt: str) -> str:
    if fmt == "jpg":
        return "jpeg"
    return fmt


if __name__ == "__main__":
    multimodal_example()
