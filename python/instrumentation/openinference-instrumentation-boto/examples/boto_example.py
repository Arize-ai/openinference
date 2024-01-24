import json

import boto3
from openinference.instrumentation.boto import BedrockInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
span_console_exporter = ConsoleSpanExporter()
span_otlp_exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:6006/v1/traces")
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=span_console_exporter))
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=span_otlp_exporter))
trace_api.set_tracer_provider(tracer_provider=tracer_provider)

BedrockInstrumentor().instrument()

session = boto3.session.Session()
client = session.client("bedrock-runtime")


if __name__ == "__main__":
    prompt = (
        b'{"prompt": "Human: Hello there, how are you? Assistant:", "max_tokens_to_sample": 1024}'
    )
    response = client.invoke_model(modelId="anthropic.claude-v2", body=prompt)
    response_body = json.loads(response.get("body").read())
    print(response_body["completion"])
