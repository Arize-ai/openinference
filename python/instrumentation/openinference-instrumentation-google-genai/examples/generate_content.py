import asyncio

from google import genai
from google.genai.types import Content, GenerateContentConfig, Part
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

endpoint = "http://0.0.0.0:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

GEMINI_API_KEY = "*REPLACE_WITH_YOUR_API_KEY*"


def generate_content_sync(model="gemini-2.0-flash"):
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        config = GenerateContentConfig(
            system_instruction="You are a helpful assistant that can answer questions and help with tasks."
        )
        content = Content(
            role="user",
            parts=[
                Part.from_text(text="Why is the sky blue?"),
                Part.from_text(text="What is the capital of France?"),
            ],
        )
        response = client.models.generate_content(model=model, contents=content, config=config)

        return response.text

    except Exception as e:
        return f"Error generating response: {str(e)}"


async def generate_content_async(model="gemini-2.0-flash"):
    try:
        client = genai.Client(api_key=GEMINI_API_KEY).aio

        config = GenerateContentConfig(
            system_instruction="You are a helpful assistant that can answer questions and help with tasks."
        )
        content = Content(
            role="user",
            parts=[
                Part.from_text(text="Why is the sky blue?"),
                Part.from_text(text="What is the capital of France?"),
            ],
        )

        async_response = await client.models.generate_content(
            model=model, contents=content, config=config
        )
        return async_response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"


if __name__ == "__main__":
    GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    response = generate_content_sync("Why is the sky blue?")
    print(response)

    async_response = asyncio.run(generate_content_async("Why is the sky blue?"))
    print(async_response)
