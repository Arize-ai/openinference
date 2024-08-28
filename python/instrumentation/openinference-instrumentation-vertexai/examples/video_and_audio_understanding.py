import vertexai
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from vertexai.generative_models import GenerativeModel, Part

from openinference.instrumentation.vertexai import VertexAIInstrumentor

endpoint = "http://127.0.0.1:4317"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

VertexAIInstrumentor().instrument(tracer_provider=tracer_provider)

vertexai.init(location="us-central1")
model = GenerativeModel("gemini-1.5-flash")
prompt = """Provide a description of the video.
The description should also contain anything important which people say in the video.
"""
video_file_uri = "gs://cloud-samples-data/generative-ai/video/pixel8.mp4"
video_file = Part.from_uri(video_file_uri, mime_type="video/mp4")
contents = [video_file, prompt]

if __name__ == "__main__":
    response = model.generate_content(contents)
    print(response)
