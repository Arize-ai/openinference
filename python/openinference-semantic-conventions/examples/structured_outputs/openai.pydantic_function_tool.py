import json
from contextlib import ExitStack
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import openai
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


class Table(str, Enum):
    orders = "orders"
    customers = "customers"
    products = "products"


class Column(str, Enum):
    id = "id"
    status = "status"
    expected_delivery_date = "expected_delivery_date"
    delivered_at = "delivered_at"
    shipped_at = "shipped_at"
    ordered_at = "ordered_at"
    canceled_at = "canceled_at"


class Operator(str, Enum):
    eq = "="
    gt = ">"
    lt = "<"
    le = "<="
    ge = ">="
    ne = "!="


class OrderBy(str, Enum):
    asc = "asc"
    desc = "desc"


class DynamicValue(BaseModel):
    column_name: str


class Condition(BaseModel):
    column: str
    operator: Operator
    value: Union[str, int, DynamicValue]


class Query(BaseModel):
    table_name: Table
    columns: list[Column]
    conditions: list[Condition]
    order_by: OrderBy


messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. The current date is August 6, "
        "2024. You help users query for the data they are looking for "
        "by calling the query function.",
    },
    {
        "role": "user",
        "content": "look up all my orders in may of last year that were fulfilled "
        "but not delivered on time",
    },
]


client = OpenAI()
with ExitStack() as stack:
    stack.enter_context(d := TemporaryDirectory())
    cass = stack.enter_context(
        vcr.use_cassette(
            Path(d.name) / Path(__file__).with_suffix(".yaml").name,
            filter_headers=["authorization"],
            ignore_localhost=True,
        )
    )
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        tools=[openai.pydantic_function_tool(Query)],
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
