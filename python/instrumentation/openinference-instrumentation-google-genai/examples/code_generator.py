import os

from google import genai
from google.genai import types
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,  # type: ignore[import-not-found]
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

endpoint = "http://0.0.0.0:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)


def run_weather_example() -> None:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents="What is the sum of the first 50 prime numbers? "
        "Generate and run code for the calculation, and make sure you get all 50.",
        config=types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution)]
        ),
    )
    print(list(response))


if __name__ == "__main__":
    run_weather_example()
