import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import vcr
from brotli import brotli
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage, ToolChoice
from openinference.instrumentation.mistralai import MistralAIInstrumentor
from openinference.semconv.trace import SpanAttributes
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

endpoint = "http://127.0.0.1:4317"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
in_memory_span_exporter = InMemorySpanExporter()
tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))

MistralAIInstrumentor().instrument(tracer_provider=tracer_provider)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "finds the weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather " "for, e.g. 'London'",
                    }
                },
                "required": ["city"],
            },
        },
    },
]
messages = [
    ChatMessage(
        content="What's the weather like in San Francisco?",
        role="user",
    )
]

client = MistralClient()
with ExitStack() as stack:
    stack.enter_context(d := TemporaryDirectory())
    cass = stack.enter_context(
        vcr.use_cassette(
            Path(d.name) / Path(__file__).with_suffix(".yaml").name,
            filter_headers=["authorization"],
            ignore_localhost=True,
        )
    )
    response = client.chat(
        model="open-mistral-nemo",
        tool_choice=ToolChoice.any,
        tools=tools,
        messages=messages,
    )
pairs = [
    {
        f"REQUEST-{i}": json.loads(cass.requests[i].body),
        f"RESPONSE-{i}": json.loads(brotli.decompress(cass.responses[i]["body"]["string"])),
    }
    for i in range(len(cass.requests))
]
with open(Path(__file__).with_suffix(".vcr.json"), "w") as f:
    json.dump(pairs, f, indent=2)
input_value = [
    json.loads(span.attributes[SpanAttributes.INPUT_VALUE])
    for span in in_memory_span_exporter.get_finished_spans()
]
with open(Path(__file__).with_suffix(f".{SpanAttributes.INPUT_VALUE}.json"), "w") as f:
    json.dump(input_value, f, indent=2)
