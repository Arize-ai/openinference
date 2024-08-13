import json
from contextlib import ExitStack
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import vcr
from brotli import brotli
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

endpoint = "http://127.0.0.1:4317"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
in_memory_span_exporter = InMemorySpanExporter()
tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

client = OpenAI()
MODEL = "gpt-4o-mini"


class UIType(str, Enum):
    div = "div"
    button = "button"
    header = "header"
    section = "section"
    field = "field"
    form = "form"


class Attribute(BaseModel):
    name: str
    value: str


class UI(BaseModel):
    type: UIType
    label: str
    children: List["UI"]
    attributes: List[Attribute]


UI.model_rebuild()  # This is required to enable recursive types


class Response(BaseModel):
    ui: UI


messages = [
    {
        "role": "system",
        "content": "You are a UI generator AI. Convert the user input into a UI.",
    },
    {"role": "user", "content": "Make a User Profile Form"},
]

with ExitStack() as stack:
    stack.enter_context(d := TemporaryDirectory())
    cass = stack.enter_context(
        vcr.use_cassette(
            Path(d.name) / Path(__file__).with_suffix(".yaml").name,
            filter_headers=["authorization"],
            ignore_localhost=True,
        )
    )
    print(
        client.beta.chat.completions.parse(
            model=MODEL,
            messages=messages,
            response_format=Response,
        )
        .choices[0]
        .message.parsed
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
