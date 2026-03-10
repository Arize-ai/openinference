import asyncio
import time

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.bedrock import BedrockInstrumentor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

BedrockInstrumentor().instrument(tracer_provider=tracer_provider)


async def run():
    agent_id = "XNW1LGJJZT"
    agent_alias_id = "K0P4LV9GPO"
    session_id = f"default-session1_{int(time.time())}"
    import aioboto3

    session = aioboto3.session.Session(
        region_name="us-east-1",
    )
    async with session.client("bedrock-agent-runtime") as client:
        attributes = dict(
            inputText="When is a good time to visit the Taj Mahal?",
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            enableTrace=True,
        )
        response = await client.invoke_agent(**attributes)
        async for event in response["completion"]:
            if "chunk" in event:
                print(event)
                chunk_data = event["chunk"]
                if "bytes" in chunk_data:
                    output_text = chunk_data["bytes"].decode("utf8")
                    print(output_text)
            elif "trace" in event:
                print(event["trace"])


if __name__ == "__main__":
    asyncio.run(run())
