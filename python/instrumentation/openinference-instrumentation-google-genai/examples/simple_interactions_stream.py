from google import genai
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,  # type: ignore[import-not-found]
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# Make sure to set the GEMINI_API_KEY environment variable


if __name__ == "__main__":
    client = genai.Client()

    stream = client.interactions.create(
        model="gemini-3-flash-preview",
        input="Explain quantum entanglement in simple terms.",
        stream=True,
    )

    for chunk in stream:
        print("@@@@@@@@@@@@@@@@@@@@@", chunk)
        # if chunk.event_type == "content.delta":
        #     if chunk.delta.type == "text":
        #         print(chunk.delta.text, end="", flush=True)
        #     elif chunk.delta.type == "thought":
        #         print(chunk.delta.thought, end="", flush=True)
        # elif chunk.event_type == "interaction.complete":
        #     print(f"\n\n--- Stream Finished ---")
        #     print(f"Total Tokens: {chunk.interaction.usage.total_tokens}")
