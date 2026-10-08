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


def reasoning_with_stream():
    client = openai.OpenAI()
    for event in client.responses.create(
        model="o4-mini",
        reasoning={"effort": "low", "summary": "auto"},
        input="Write a one-sentence bedtime story about a unicorn.",
        include=["reasoning.encrypted_content"],
        stream=True,
    ):
        print(event)


def reasoning_responses():
    client = openai.OpenAI()
    response = client.responses.create(
        model="o4-mini",
        reasoning={"effort": "low", "summary": "auto"},
        input="Write a one-sentence bedtime story about a unicorn.",
        include=["reasoning.encrypted_content"],
    )
    print(response)


if __name__ == "__main__":
    reasoning_responses()
    reasoning_with_stream()
