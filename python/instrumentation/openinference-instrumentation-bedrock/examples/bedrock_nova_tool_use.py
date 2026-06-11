import json
from typing import Any, Dict, List

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

client = boto3.client("bedrock-runtime", region_name="us-east-1")

MODEL_ID = "amazon.nova-micro-v1:0"

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_weather",
                "description": "Get the current weather for a given location.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. Seattle, WA.",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit.",
                            },
                        },
                        "required": ["location"],
                    }
                },
            }
        }
    ]
}


def get_weather(location: str, unit: str = "fahrenheit") -> Dict[str, Any]:
    """Return mock weather data for any location."""
    return {
        "location": location,
        "temperature": 58,
        "unit": unit,
        "condition": "Partly cloudy",
        "humidity": 72,
    }


def invoke(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    request_body: Dict[str, Any] = {
        "schemaVersion": "messages-v1",
        "messages": messages,
        "toolConfig": TOOL_CONFIG,
        "inferenceConfig": {"maxTokens": 512, "temperature": 0.1},
    }
    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(request_body),
    )
    return json.loads(response["body"].read())


def nova_multi_turn_tool_use() -> None:
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"text": "What is the weather like in Seattle, WA?"}]}
    ]

    response_body = invoke(messages)
    assistant_message = response_body["output"]["message"]
    tool_use_block = next(
        block["toolUse"] for block in assistant_message["content"] if "toolUse" in block
    )
    tool_use_id = tool_use_block["toolUseId"]
    tool_name = tool_use_block["name"]
    tool_input = tool_use_block["input"]
    print(f"  Tool called : {tool_name}({tool_input})")
    tool_result = get_weather(**tool_input)
    print(f"  Tool result : {tool_result}")

    messages.append(assistant_message)
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"json": tool_result}],
                    }
                }
            ],
        }
    )
    response_body2 = invoke(messages)

    final_text = response_body2["output"]["message"]["content"][0]["text"]
    print(f"  Final answer: {final_text}")


if __name__ == "__main__":
    nova_multi_turn_tool_use()
