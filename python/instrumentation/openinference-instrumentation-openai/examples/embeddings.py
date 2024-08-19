import openai
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.openai import OpenAIInstrumentor

resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
span_exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:6006/v1/traces")
span_processor = SimpleSpanProcessor(span_exporter=span_exporter)
tracer_provider.add_span_processor(span_processor=span_processor)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


if __name__ == "__main__":
    response = openai.OpenAI().embeddings.create(
        model="text-embedding-ada-002",
        input="hello world",
    )
    print(response.data[0].embedding)
