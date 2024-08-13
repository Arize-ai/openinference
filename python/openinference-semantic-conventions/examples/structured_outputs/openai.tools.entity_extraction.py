import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import vcr
from brotli import brotli
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
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

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

client = OpenAI()
MODEL = "gpt-4o-mini"

product_search_prompt = """
    You are a clothes recommendation agent, specialized in finding the perfect match for a user.
    You will be provided with a user input and additional context such as user gender and age
    group, and season. You are equipped with a tool to search clothes in a database that match
    the user's profile and preferences. Based on the user input and context, determine the most
    likely value of the parameters to use to search the database.

    Here are the different categories that are available on the website:
    - shoes: boots, sneakers, sandals
    - jackets: winter coats, cardigans, parkas, rain jackets
    - tops: shirts, blouses, t-shirts, crop tops, sweaters
    - bottoms: jeans, skirts, trousers, joggers

    There are a wide range of colors available, but try to stick to regular color names.
"""

product_search_function = {
    "type": "function",
    "function": {
        "name": "product_search",
        "description": "Search for a match in the product database",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The broad category of the product",
                    "enum": ["shoes", "jackets", "tops", "bottoms"],
                },
                "subcategory": {
                    "type": "string",
                    "description": "The sub category of the product, within the broader category",
                },
                "color": {
                    "type": "string",
                    "description": "The color of the product",
                },
            },
            "required": ["category", "subcategory", "color"],
            "additionalProperties": False,
        },
    },
    "strict": True,
}


def get_response(user_input, context):
    messages = [
        {"role": "system", "content": product_search_prompt},
        {"role": "user", "content": f"CONTEXT: {context}\n USER INPUT: {user_input}"},
    ]
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=messages,
        tools=[product_search_function],
    )

    return response.choices[0].message.tool_calls


example_inputs = [
    {
        "user_input": "I'm looking for a new coat. I'm always cold so please "
        "something warm! Ideally something that matches my eyes.",
        "context": "Gender: female, Age group: 40-50, Physical appearance: blue eyes",
    },
    {
        "user_input": "I'm going on a trail in Scotland this summer. It's "
        "going to be rainy. Help me find something.",
        "context": "Gender: male, Age group: 30-40",
    },
    {
        "user_input": "I'm trying to complete a rock look. I'm missing shoes. Any suggestions?",
        "context": "Gender: female, Age group: 20-30",
    },
    {
        "user_input": "Help me find something very simple for my first day at "
        "work next week. Something casual and neutral.",
        "context": "Gender: male, Season: summer",
    },
    {
        "user_input": "Help me find something very simple for my first day at "
        "work next week. Something casual and neutral.",
        "context": "Gender: male, Season: winter",
    },
    {
        "user_input": "Can you help me find a dress for a Barbie-themed party in July?",
        "context": "Gender: female, Age group: 20-30",
    },
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
    for ex in example_inputs:
        ex["result"] = get_response(ex["user_input"], ex["context"])
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
