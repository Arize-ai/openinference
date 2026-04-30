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
client = session.client(
    "bedrock-runtime",
    "us-east-1",
)


def get_top_song(sing: str):
    return {"title": "Never Gonna Give You Up", "author": "Rick Astley", "played_times": 420}


tool_config = {
    "tools": [
        {
            "toolSpec": {
                "name": "top_song",
                "description": "Get the most popular song played on a radio station.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "sign": {
                                "type": "string",
                                "description": "The call sign for the radio station.",
                            }
                        },
                        "required": ["sign"],
                    }
                },
            }
        }
    ]
}


def converse_example():
    system_prompt = [{"text": "You are an expert at creating music playlists"}]
    messages = [
        {"role": "user", "content": [{"text": "What is the most popular song on Radio XYZ?"}]}
    ]
    model_id = "mistral.devstral-2-123b"
    inference_config = {"maxTokens": 1024, "temperature": 0.0}
    response = client.converse(
        modelId=model_id,
        system=system_prompt,
        toolConfig=tool_config,
        messages=messages,
        inferenceConfig=inference_config,
    )
    tool_request = response["output"]["message"]["content"][0]["toolUse"]
    messages.append(response["output"]["message"])
    tool_result = get_top_song(tool_request["input"]["sign"])
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": tool_request["toolUseId"],
                        "content": [{"json": tool_result}],
                    }
                }
            ],
        }
    )

    final_response = client.converse(
        modelId=model_id,
        system=system_prompt,
        toolConfig=tool_config,
        messages=messages,
        inferenceConfig=inference_config,
    )
    print(final_response["output"]["message"]["content"][0]["text"])


if __name__ == "__main__":
    converse_example()
