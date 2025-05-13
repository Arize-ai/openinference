import logging
import sys

import dspy
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.dspy import DSPyInstrumentor

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
span_console_exporter = ConsoleSpanExporter()
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=span_console_exporter))

endpoint = "http://localhost:6006/v1/traces"
span_otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=span_otlp_exporter))


trace_api.set_tracer_provider(tracer_provider=tracer_provider)
DSPyInstrumentor().instrument()

gpt4o = dspy.LM("openai/gpt-4o", temperature=0.7)
dspy.configure(lm=gpt4o)


def add(x: int, y: int) -> int:
    return x + y


react = dspy.ReAct("question -> answer", tools=[add])
question = "What is 2 + 2?"

print(react(question=question))
