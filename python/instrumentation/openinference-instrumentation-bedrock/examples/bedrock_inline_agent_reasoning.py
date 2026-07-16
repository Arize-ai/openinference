import json
import time

import boto3
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.bedrock import BedrockInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace_api.set_tracer_provider(tracer_provider=tracer_provider)

BedrockInstrumentor().instrument()

session = boto3.session.Session()
client = session.client("bedrock-agent-runtime", "us-east-1")

MODEL_ID = "openai.gpt-oss-120b-1:0"
# MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def simple_reasoning_agent() -> None:
    session_id = f"inline-reasoning-session_{int(time.time())}"

    response = client.invoke_inline_agent(
        foundationModel=MODEL_ID,
        instruction=(
            "You are a helpful assistant. Think through problems step by step before answering."
        ),
        inputText="What is the 10th Fibonacci number? Think step by step.",
        sessionId=session_id,
        enableTrace=True,
    )
    print(json.dumps(list(response["completion"]), default=str))


if __name__ == "__main__":
    simple_reasoning_agent()
