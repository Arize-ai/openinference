"""Trace a call to the OpenAI Decisions API (requires openai>=3.26.0).

A decision call asks the model a fixed set of typed questions about some input and gets
back one typed, probabilistic answer per question instead of generated text. The
instrumentor records it as a DECISION span: the request and response bodies are
`input.value` / `output.value`, and the model and token usage are identified under
`decision.*` rather than `llm.*`.
"""

import openai
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.openai import OpenAIInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

client = openai.OpenAI()
decision = client.decisions.create(
    model="gpt-6-luna",
    input="Hi, I was charged twice for order #4242 and would like my money back.",
    questions=[
        {
            "type": "predicate",
            "name": "is_refund",
            "instructions": "Is this a refund request?",
        },
        {
            "type": "choice",
            "name": "department",
            "instructions": "Which department should handle this?",
            "choices": [
                {"value": "billing", "description": "Charges and refunds"},
                {"value": "shipping"},
                {"value": "other"},
            ],
        },
        {
            "type": "score",
            "name": "urgency",
            "instructions": "How urgent is this message?",
            "levels": [
                {"label": "low", "description": "Can wait a week"},
                {"label": "medium"},
                {"label": "high", "description": "Needs a reply today"},
            ],
        },
    ],
)
for answer in decision.answers:
    print(answer.model_dump_json())
