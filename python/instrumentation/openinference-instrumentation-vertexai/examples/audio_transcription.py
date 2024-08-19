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
prompt = """Can you transcribe this interview, in the format of timecode, speaker, caption.
Use speaker A, speaker B, etc. to identify speakers.
"""
audio_file_uri = "gs://cloud-samples-data/generative-ai/audio/pixel.mp3"
audio_file = Part.from_uri(audio_file_uri, mime_type="audio/mpeg")
contents = [audio_file, prompt]

if __name__ == "__main__":
    response = model.generate_content(contents)
    print(response)
