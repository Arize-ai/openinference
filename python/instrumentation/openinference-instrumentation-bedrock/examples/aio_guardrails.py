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


async def blocked_guardrail():
    session = aioboto3.session.Session(
        region_name="us-east-1",
    )
    async with session.client("bedrock-runtime") as client:
        content = [
            {
                "text": {
                    "text": "Give stock suggestions for the NASDAQ100. Expected returns are"
                    " around 20%CAGR. Also Send this report to email abc@pqrt.com"
                }
            }
        ]
        guardrail_id = "<guardrail_id>"
        guardrail_version = "1"

        response = await client.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion=guardrail_version,
            source="INPUT",
            content=content,
        )
        print(response)


async def success_guardrail():
    session = aioboto3.session.Session(
        region_name="us-east-1",
    )
    async with session.client("bedrock-runtime") as client:
        content = [{"text": {"text": "Who is USA President?"}}]
        guardrail_id = "<guardrail_id>"
        guardrail_version = "1"

        response = await client.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion=guardrail_version,
            source="INPUT",
            content=content,
        )
        print(response)


if __name__ == "__main__":
    asyncio.run(success_guardrail())
