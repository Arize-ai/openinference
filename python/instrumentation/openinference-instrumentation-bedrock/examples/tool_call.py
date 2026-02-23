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

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
tools = [
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol, e.g. AAPL for Apple Inc.",
                }
            },
            "required": ["ticker"],
        },
    }
]


def tool_call_example():
    # Prepare Claude messages payload
    body = {
        "messages": [{"role": "user", "content": "What's the S&P 500 at today?"}],
        "tools": tools,
        "max_tokens": 1000,
        "temperature": 0.7,
        "anthropic_version": "bedrock-2023-05-31",
    }

    # Send request to Claude via Bedrock
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    response_body = json.loads(response["body"].read())

    # Display the response
    print("Claude response:")
    print(json.dumps(response_body, indent=2))


def tool_call_with_response():
    # Prepare Claude messages payload
    body = {
        "messages": [
            {"role": "user", "content": "What's the S&P 500 at today?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
                        "name": "get_stock_price",
                        "input": {"ticker": "^GSPC"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
                        "content": "259.75 USD",
                    }
                ],
            },
        ],
        "tools": tools,
        "max_tokens": 1000,
        "temperature": 0.7,
        "anthropic_version": "bedrock-2023-05-31",
    }

    # Send request to Claude via Bedrock
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    response_body = json.loads(response["body"].read())

    # Display the response
    print("Claude response:")
    print(json.dumps(response_body, indent=2))


if __name__ == "__main__":
    tool_call_example()
    tool_call_with_response()
