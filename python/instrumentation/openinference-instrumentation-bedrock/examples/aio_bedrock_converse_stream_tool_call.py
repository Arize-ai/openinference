import asyncio

import aioboto3
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.bedrock import BedrockInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

BedrockInstrumentor().instrument(tracer_provider=tracer_provider)

tool_config = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_stock_price",
                "description": "Get the current stock price for a given ticker symbol.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The stock ticker symbol, e.g. AAPL for Apple Inc.",
                            }
                        },
                        "required": ["ticker"],
                    }
                },
            }
        }
    ]
}


async def tool_call_example():
    session = aioboto3.session.Session(
        region_name="us-east-1",
    )
    async with session.client("bedrock-runtime") as client:
        messages = [{"role": "user", "content": [{"text": "What's the S&P 500 at today?"}]}]
        response = await client.converse_stream(
            modelId="us.anthropic.claude-sonnet-4-6",
            messages=messages,
            toolConfig=tool_config,
            inferenceConfig={"maxTokens": 1000, "temperature": 0.7},
        )
        print("Streaming response (tool call):")
        async for event in response["stream"]:
            print(event)


async def tool_call_with_response():
    session = aioboto3.session.Session(
        region_name="us-east-1",
    )
    async with session.client("bedrock-runtime") as client:
        messages = [
            {"role": "user", "content": [{"text": "What's the S&P 500 at today?"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_01D7FLrfh4GYq7yT1ULFeyMV",
                            "name": "get_stock_price",
                            "input": {"ticker": "^GSPC"},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tooluse_01D7FLrfh4GYq7yT1ULFeyMV",
                            "content": [{"text": "259.75 USD"}],
                        }
                    }
                ],
            },
        ]
        response = await client.converse_stream(
            modelId="us.anthropic.claude-sonnet-4-6",
            messages=messages,
            toolConfig=tool_config,
            inferenceConfig={"maxTokens": 1000, "temperature": 0.7},
        )
        print("Streaming response (tool result):")
        async for event in response["stream"]:
            print(event)


async def main():
    await tool_call_example()
    await tool_call_with_response()


if __name__ == "__main__":
    asyncio.run(main())
