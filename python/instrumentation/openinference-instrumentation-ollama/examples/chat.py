import os

import ollama
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.ollama import OllamaInstrumentor
from openinference.semconv.resource import ResourceAttributes

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider(
    resource=Resource.create({ResourceAttributes.PROJECT_NAME: "ollama-examples"})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

OllamaInstrumentor().instrument(tracer_provider=tracer_provider)

MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

if __name__ == "__main__":
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
    )
    print(response.message.content)
