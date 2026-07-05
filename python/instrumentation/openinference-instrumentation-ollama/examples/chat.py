import ollama
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.ollama import OllamaInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

OllamaInstrumentor().instrument(tracer_provider=tracer_provider)


if __name__ == "__main__":
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
    )
    print(response.message.content)
