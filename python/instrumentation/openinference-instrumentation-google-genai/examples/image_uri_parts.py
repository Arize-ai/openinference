from google import genai
from google.genai.types import Part
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,  # type: ignore[import-not-found]
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

endpoint = "http://0.0.0.0:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)
MODEL_ID = "gemini-3-flash-preview"

if __name__ == "__main__":
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "What is shown in this image?",
            Part.from_uri(
                file_uri="https://fastly.picsum.photos/id/237/200/300.jpg?hmac=TmmQSbShHz9CdQm0NkEjx1Dyh_Y984R9LpNrpvH2D_U",
                mime_type="image/jpeg",
            ),
        ],
    )

    print(response.text)
