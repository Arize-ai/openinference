import json

import boto3
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation import using_attributes
from openinference.instrumentation.bedrock import BedrockInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace_api.set_tracer_provider(tracer_provider=tracer_provider)

BedrockInstrumentor().instrument()

client = boto3.client("bedrock-runtime", region_name="us-east-1")


def nova_invoke_model() -> None:
    request_body = {
        "schemaVersion": "messages-v1",
        "messages": [{"role": "user", "content": [{"text": "What are the three primary colors?"}]}],
        "inferenceConfig": {"maxTokens": 256, "temperature": 0.7},
    }
    response = client.invoke_model(
        modelId="amazon.nova-micro-v1:0",
        body=json.dumps(request_body),
    )
    response_body = json.loads(response["body"].read())
    output_text = response_body["output"]["message"]["content"][0]["text"]
    print(f"Nova Micro response: {output_text}")
    print(f"Token usage: {response_body['usage']}")


def nova_invoke_model_stream() -> None:
    request_body = {
        "schemaVersion": "messages-v1",
        "messages": [{"role": "user", "content": [{"text": "What are the three primary colors?"}]}],
        "inferenceConfig": {"maxTokens": 256, "temperature": 0.7},
    }
    response = client.invoke_model_with_response_stream(
        modelId="amazon.nova-micro-v1:0",
        body=json.dumps(request_body),
    )
    response_body = list(response["body"])
    print(f"Response: {response_body}")


def nova_converse_with_multi_turn() -> None:
    messages = []
    inference_config = {"maxTokens": 512, "temperature": 0.5}

    with using_attributes(
        session_id="nova-demo-session",
        user_id="demo-user",
        tags=["nova", "converse"],
    ):
        messages.append(
            {
                "role": "user",
                "content": [{"text": "Name three famous scientists and their fields."}],
            }
        )
        response = client.converse(
            modelId="amazon.nova-lite-v1:0",
            system=[{"text": "You are a knowledgeable assistant. Keep answers concise."}],
            messages=messages,
            inferenceConfig=inference_config,
        )
        assistant_msg = response["output"]["message"]
        print(f"Turn 1: {assistant_msg['content'][0]['text']}")
        messages.append(assistant_msg)
        messages.append({"role": "user", "content": [{"text": "Which of them worked on physics?"}]})
        response = client.converse(
            modelId="amazon.nova-lite-v1:0",
            system=[{"text": "You are a knowledgeable assistant. Keep answers concise."}],
            messages=messages,
            inferenceConfig=inference_config,
        )
        assistant_msg = response["output"]["message"]
        print(f"Turn 2: {assistant_msg['content'][0]['text']}")


if __name__ == "__main__":
    nova_invoke_model_stream()
    # nova_invoke_model()
    # nova_converse_with_multi_turn()
