from dspy import LM
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.dspy import DSPyInstrumentor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
DSPyInstrumentor().instrument(tracer_provider=tracer_provider)


if __name__ == "__main__":
    lm = LM("openai/gpt-4.1")
    output = lm("hello!")
    print(f"{output=}")
    DSPyInstrumentor().uninstrument()
    output = lm("hello! What is latest on AI?")
    print(output)
