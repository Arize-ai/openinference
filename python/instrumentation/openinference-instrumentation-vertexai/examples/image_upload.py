from io import BytesIO

import PIL
import requests
import vertexai
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from vertexai.generative_models import GenerativeModel, Image, Part

from openinference.instrumentation import TraceConfig
from openinference.instrumentation.vertexai import VertexAIInstrumentor

endpoint = "http://127.0.0.1:4317"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

VertexAIInstrumentor().instrument(
    tracer_provider=tracer_provider,
    config=TraceConfig(base64_image_max_length=200_000),
)
vertexai.init(location="us-central1")
model = GenerativeModel("gemini-1.5-flash")

url = "https://nextml.github.io/caption-contest-data/cartoons/893.jpg"
image = PIL.Image.open(BytesIO(requests.get(url).content))
image.thumbnail((512, 512))
arr = BytesIO()
image.save(arr, format=image.format)

if __name__ == "__main__":
    response = model.generate_content(
        [
            "What's funny about this caption?\n\nThe seller isn't willing to come down.",
            Part.from_image(Image.from_bytes(arr.getvalue())),
        ]
    )
    print(response)
