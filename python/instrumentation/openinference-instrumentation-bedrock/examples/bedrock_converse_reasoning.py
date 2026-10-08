import boto3
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.bedrock import BedrockInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
trace_api.set_tracer_provider(tracer_provider=tracer_provider)


BedrockInstrumentor().instrument()
region = "us-east-1"
session = boto3.session.Session()
client = session.client("bedrock-runtime", region_name=region)

# MODEL_ID = "openai.gpt-oss-120b-1:0"
MODEL_ID = "us.anthropic.claude-sonnet-4-6"


def converse_with_reasoning() -> None:
    response = client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [{"text": "What is the 10th Fibonacci number? Think step by step."}],
            }
        ],
        inferenceConfig={"maxTokens": 16000},
        additionalModelRequestFields={"thinking": {"type": "enabled", "budget_tokens": 5000}},
    )
    print(response)


def converse_stream_with_reasoning() -> None:
    response = client.converse_stream(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [{"text": "What is the 10th Fibonacci number? Think step by step."}],
            }
        ],
        inferenceConfig={"maxTokens": 16000},
        additionalModelRequestFields={"thinking": {"type": "enabled", "budget_tokens": 5000}},
    )
    events = list(response["stream"])
    print(events)


if __name__ == "__main__":
    converse_with_reasoning()
    converse_stream_with_reasoning()
