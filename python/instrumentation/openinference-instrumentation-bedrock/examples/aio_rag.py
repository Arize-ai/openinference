import asyncio

import aioboto3
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


async def run():
    session = aioboto3.session.Session(
        region_name="us-east-1",
    )
    attributes = dict(
        knowledgeBaseId="QKERWOBDH0", retrievalQuery={"text": "What is task Decomposition?"}
    )
    async with session.client("bedrock-agent-runtime") as client:
        response = await client.retrieve(**attributes)
        print(response)


async def run_generate():
    session = aioboto3.session.Session(
        region_name="us-east-1",
    )

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
    async with session.client("bedrock-agent-runtime") as client:
        response = await client.retrieve_and_generate_stream(**attributes)
        async for res in response['stream']:
            print(res)


if __name__ == "__main__":
    asyncio.run(run())
    asyncio.run(run_generate())
