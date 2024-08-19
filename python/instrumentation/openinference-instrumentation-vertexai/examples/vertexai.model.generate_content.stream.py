import vertexai
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from vertexai.generative_models import GenerativeModel

from openinference.instrumentation.vertexai import VertexAIInstrumentor

endpoint = "http://127.0.0.1:4317"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

VertexAIInstrumentor().instrument(tracer_provider=tracer_provider)

vertexai.init(location="us-central1")
model = GenerativeModel("gemini-1.5-flash")


if __name__ == "__main__":
    response_gen = model.generate_content(
        "Write a haiku.",
        generation_config={"max_output_tokens": 20},
        stream=True,
    )
    for response in response_gen:
        # print(response.text, end="")
        print(response.candidates[0].content.parts[0]._raw_part.text, end="")
