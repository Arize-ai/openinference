import json

import boto3
from openinference.instrumentation import using_attributes
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
client = session.client("bedrock-runtime", region_name="us-east-1")

def invoke_example():
    prompt = (
        b'{"prompt": "Human: Hello there, how are you? Assistant:", "max_tokens_to_sample": 1024}'
    )
    with using_attributes(
        session_id="my-test-session",
        user_id="my-test-user",
        metadata={
            "test-int": 1,
            "test-str": "string",
            "test-list": [1, 2, 3],
            "test-dict": {
                "key-1": "val-1",
                "key-2": "val-2",
            },
        },
        tags=["tag-1", "tag-2"],
        prompt_template="Who won the soccer match in {city} on {date}",
        prompt_template_version="v1.0",
        prompt_template_variables={
            "city": "Johannesburg",
            "date": "July 11th",
        },
    ):
        response = client.invoke_model(modelId="anthropic.claude-v2", body=prompt)
    response_body = json.loads(response.get("body").read())
    print(response_body["completion"])

def converse_example():
    message1 = {
            "role": "user",
            "content": [{"text": "hello there?"}]
    }
    message2 = {
            "role": "user",
            "content": [{"text": "Make sure the songs are by artists from the United Kingdom."}]
    }
    messages = []

    with using_attributes(
        session_id="my-test-session",
        user_id="my-test-user",
        metadata={
            "test-int": 1,
            "test-str": "string",
            "test-list": [1, 2, 3],
            "test-dict": {
                "key-1": "val-1",
                "key-2": "val-2",
            },
        },
        tags=["tag-1", "tag-2"],
        prompt_template="Who won the soccer match in {city} on {date}",
        prompt_template_version="v1.0",
        prompt_template_variables={
            "city": "Johannesburg",
            "date": "July 11th",
        },
    ):
        messages.append(message1)
        response = client.converse(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages=messages
        )
        out = response['output']['message']
        messages.append(out)
        print(out.get("content")[-1].get("text"))

        messages.append(message2)
        response = client.converse(
            modelId="anthropic.claude-v2:1",
            messages=messages
        )
        out = response['output']['message']
        print(response)
        print(out.get("content", {}).get("text"))

if __name__ == "__main__":
    invoke_example()
    converse_example()
