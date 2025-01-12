import os

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
)
from smolagents import OpenAIServerModel

from openinference.instrumentation.smolagents import SmolagentsInstrumentor

endpoint = "http://0.0.0.0:6006/v1/traces"
trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider, skip_dep_check=True)

model = OpenAIServerModel(
    model_id="gpt-4o", api_key=os.environ["OPENAI_API_KEY"], api_base="https://api.openai.com/v1"
)
output = model(messages=[{"role": "user", "content": "hello world"}])
print(output)
