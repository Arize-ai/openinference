import json

import boto3
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.bedrock import BedrockInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
trace_api.set_tracer_provider(tracer_provider=tracer_provider)

BedrockInstrumentor().instrument()

session = boto3.session.Session()
client = session.client("bedrock-runtime", "us-east-1")

MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
SYSTEM_PROMPT = "You are a pirate. Always answer in one short sentence."


def request_body(system):
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "system": system,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    }


def invoke_model_with_string_system():
    response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(request_body(SYSTEM_PROMPT)))
    response_body = json.loads(response.get("body").read())
    print(response_body["content"][0]["text"])


def invoke_model_with_block_system():
    system = [{"type": "text", "text": SYSTEM_PROMPT}]
    response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(request_body(system)))
    response_body = json.loads(response.get("body").read())
    print(response_body["content"][0]["text"])


def invoke_model_with_response_stream_with_block_system():
    system = [{"type": "text", "text": SYSTEM_PROMPT}]
    response = client.invoke_model_with_response_stream(
        modelId=MODEL_ID, body=json.dumps(request_body(system))
    )
    for event in response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        if chunk["type"] == "content_block_delta" and "text" in chunk["delta"]:
            print(chunk["delta"]["text"], end="")
    print()


if __name__ == "__main__":
    invoke_model_with_string_system()
    invoke_model_with_block_system()
    invoke_model_with_response_stream_with_block_system()
