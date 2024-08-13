import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import vcr
from brotli import brotli
from IPython.display import Math, display
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

math_tutor_prompt = """
You are a helpful math tutor. You will be provided with a math problem,
and your goal will be to output a step by step solution, along with a final answer.
For each step, just provide the output as an equation use the explanation field
to detail the reasoning.
"""
question = "how can I solve 8x + 7 = -23"
messages = [
    {"role": "system", "content": math_tutor_prompt},
    {"role": "user", "content": question},
]

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "math_reasoning",
        "schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "explanation": {"type": "string"},
                            "output": {"type": "string"},
                        },
                        "required": ["explanation", "output"],
                        "additionalProperties": False,
                    },
                },
                "final_answer": {"type": "string"},
            },
            "required": ["steps", "final_answer"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


class MathReasoning(BaseModel):
    class Step(BaseModel):
        explanation: str
        output: str

    steps: list[Step]
    final_answer: str


def print_math_response(response):
    result = json.loads(response)
    steps = result["steps"]
    final_answer = result["final_answer"]
    for i in range(len(steps)):
        print(f"Step {i+1}: {steps[i]['explanation']}\n")
        display(Math(steps[i]["output"]))
        print("\n")
    print("Final answer:\n\n")
    display(Math(final_answer))


with ExitStack() as stack:
    stack.enter_context(d := TemporaryDirectory())
    cass = stack.enter_context(
        vcr.use_cassette(
            Path(d.name) / Path(__file__).with_suffix(".yaml").name,
            filter_headers=["authorization"],
            decode_compressed_response=True,
            ignore_localhost=True,
        )
    )
    result = (
        client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format=response_format,
        )
        .choices[0]
        .message
    )
    print_math_response(result.content)
    result = (
        client.beta.chat.completions.parse(
            model=MODEL,
            messages=messages,
            response_format=MathReasoning,
        )
        .choices[0]
        .message.parsed
    )
    print(result.steps)
    print("Final answer:")
    print(result.final_answer)
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
