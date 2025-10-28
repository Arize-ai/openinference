import litellm
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.litellm import LiteLLMInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
resource = Resource.create(
    {
        "service.name": "litellm-openai-example",
        "openinference.project.name": "openinference-litellm-demo",
    }
)
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)


if __name__ == "__main__":
    litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"content": "What's the capital of China?", "role": "user"}],
    )
    litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[
            {"content": "Hello, I want to bake a cake", "role": "user"},
            {"content": "Hello, I can pull up some recipes for cakes.", "role": "assistant"},
            {"content": "No actually I want to make a pie", "role": "user"},
        ],
        temperature=0.7,
    )
