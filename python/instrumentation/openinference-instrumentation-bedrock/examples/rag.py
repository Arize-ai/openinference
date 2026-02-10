import boto3
from opentelemetry import trace as trace_api
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
trace_api.set_tracer_provider(tracer_provider=tracer_provider)

BedrockInstrumentor().instrument()

client = boto3.client(
    "bedrock-agent-runtime",
    region_name="us-east-1",
)


def run():
    attributes = dict(
        knowledgeBaseId="QKERWOBDH0", retrievalQuery={"text": "What is task Decomposition?"}
    )
    response = client.retrieve(**attributes)
    print(response)


def retrieve_and_generate():
    attributes = {
        "input": {"text": "What is Task Decomposition?"},
        "retrieveAndGenerateConfiguration": {
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": "QKERWOBDH0",
                "modelArn": "anthropic.claude-3-haiku-20240307-v1:0",
            },
            "type": "KNOWLEDGE_BASE",
        },
    }
    response = client.retrieve_and_generate(**attributes)
    print(response)


if __name__ == "__main__":
    run()
    retrieve_and_generate()
