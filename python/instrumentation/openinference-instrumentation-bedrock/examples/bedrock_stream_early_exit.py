"""
Streams from converse_stream and invoke_model_with_response_stream and stops
reading in different ways: reading to the end, breaking out of the loop,
calling close(), and dropping the stream.

Every span should end and show up in Phoenix. The script also prints how many
spans are still open after each case.
"""

import gc
import json

import boto3
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation import using_metadata
from openinference.instrumentation.bedrock import BedrockInstrumentor


class OpenSpans(SpanProcessor):
    def __init__(self) -> None:
        self.open = set()

    def on_start(self, span, parent_context=None) -> None:
        self.open.add(span.context.span_id)

    def on_end(self, span) -> None:
        self.open.discard(span.context.span_id)


endpoint = "http://127.0.0.1:6006/v1/traces"
resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
open_spans = OpenSpans()
tracer_provider.add_span_processor(open_spans)
trace_api.set_tracer_provider(tracer_provider=tracer_provider)

BedrockInstrumentor().instrument()

session = boto3.session.Session()
client = session.client("bedrock-runtime", "us-east-1")

MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
PROMPT = "Count from one to twenty in words, one per line."


def converse_stream():
    response = client.converse_stream(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": PROMPT}]}],
        inferenceConfig={"maxTokens": 200},
    )
    return response["stream"]


def invoke_model_with_response_stream():
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
        }
    )
    response = client.invoke_model_with_response_stream(modelId=MODEL_ID, body=body)
    return response["body"]


def read_all(open_stream):
    for _ in open_stream():
        pass


def break_early(open_stream):
    for i, _ in enumerate(open_stream()):
        if i == 2:
            break


def close_early(open_stream):
    stream = open_stream()
    events = iter(stream)
    for _ in range(3):
        next(events)
    stream.close()


def drop_early(open_stream):
    stream = open_stream()
    events = iter(stream)
    for _ in range(3):
        next(events)
    del events, stream


if __name__ == "__main__":
    for open_stream in (converse_stream, invoke_model_with_response_stream):
        print(f"--- {open_stream.__name__}")
        for case in (read_all, break_early, close_early, drop_early):
            open_spans.open.clear()
            with using_metadata({"api": open_stream.__name__, "case": case.__name__}):
                case(open_stream)
            gc.collect()
            print(f"  {case.__name__:12s} spans still open: {len(open_spans.open)}")
    tracer_provider.force_flush()
