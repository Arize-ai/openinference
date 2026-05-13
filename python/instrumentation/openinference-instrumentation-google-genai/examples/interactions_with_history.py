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
    store_enabled = False
    conversation_history = [
        {
            "type": "user_input",
            "content": [{"type": "text", "text": "What are the three largest cities in Spain?"}],
        }
    ]

    interaction1 = client.interactions.create(
        model=model_id,
        store=store_enabled,
        input=conversation_history,
    )
    print(f"Response 1: {interaction1.steps[-1].content[0].text}")

    # Append the response steps from model to conversation history
    for step in interaction1.steps:
        # Convert the SDK Step object to a dictionary
        conversation_history.append(step.model_dump())

    conversation_history.append(
        {
            "type": "user_input",
            "content": [
                {"type": "text", "text": "What is the most famous landmark in the second one?"}
            ],
        }
    )

    interaction2 = client.interactions.create(
        model=model_id,
        store=store_enabled,
        input=conversation_history,
    )
    print(f"Response 2: {interaction2.steps[-1].content[0].text}")
