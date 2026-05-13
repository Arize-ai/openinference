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
    model_id = "gemini-3-flash-preview"

    # Turn 1: Initial Request
    # The 'input' can be a simple string or a structured step_list
    interaction1 = client.interactions.create(
        model=model_id, 
        input="What are the three largest cities in Spain?"
    )

    # Note: Accessing content now requires checking step types
    # Typically the last step is 'model_output'
    response_text1 = interaction1.steps[-1].content[0].text
    print(f"Model 1: {response_text1}")

    # Turn 2: Follow-up using 'previous_interaction_id'
    # This avoids the 'turn_list' error by letting the server handle history.
    interaction2 = client.interactions.create(
        model=model_id,
        input="What is the most famous landmark in the second one?",
        previous_interaction_id=interaction1.id  # Server-side history management
    )

    response_text2 = interaction2.steps[-1].content[0].text
    print(f"Model 2: {response_text2}")
