from google import genai
from google.genai.types import Content, GenerateContentConfig, Part
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

GEMINI_API_KEY = "AIzaSyA2NSDWUdii0ylHPhHO6qtuh9qvvK7PYQI"

endpoint = "http://0.0.0.0:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))


def generate_ai_response(prompt, model="gemini-2.0-flash"):
    """
    Generates AI response using Google's Gemini API.

    Args:
        prompt (str): The input prompt to send to the model
        api_key (str): Google API key for authentication
        model (str): Model name to use, defaults to gemini-2.0-flash

    Returns:
        str: The generated response text
    """
    try:
        # Initialize the client
        client = genai.Client(api_key=GEMINI_API_KEY)

        config = GenerateContentConfig(
            system_instruction="You are a helpful assistant that can answer questions and help with tasks."
        )
        content = Content(parts=[Part(text=prompt)])
        # Generate the response
        response = client.models.generate_content(model=model, contents=content, config=config)

        return response.text

    except Exception as e:
        return f"Error generating response: {str(e)}"


if __name__ == "__main__":
    GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    response = generate_ai_response("Why is the sky blue?")
    print(response)
