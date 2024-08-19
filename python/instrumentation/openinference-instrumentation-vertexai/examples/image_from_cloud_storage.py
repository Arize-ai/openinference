import vertexai
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from vertexai.generative_models import GenerativeModel, Part

from openinference.instrumentation.vertexai import VertexAIInstrumentor

endpoint = "http://127.0.0.1:4317"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

VertexAIInstrumentor().instrument(tracer_provider=tracer_provider)

vertexai.init(location="us-central1")
model = GenerativeModel("gemini-1.5-flash")

if __name__ == "__main__":
    image_part = Part.from_uri(
        "gs://download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg",
        mime_type="image/jpeg",
    )
    response = model.generate_content(["Describe this image", image_part])
    print(response)
