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
    async with session.client("bedrock-runtime") as client:
        system_prompt = [{"text": "You are an expert at creating music playlists"}]
        user_message = {"role": "user", "content": [{"text": "Create a list of 3 pop songs."}]}
        inference_config = {"maxTokens": 1024, "temperature": 0.0}
        messages = [user_message]
        response = await client.converse_stream(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            system=system_prompt,
            messages=messages,
            inferenceConfig=inference_config,
        )
        async for stream in response["stream"]:
            print(stream)


if __name__ == "__main__":
    asyncio.run(run())
