import asyncio
import os

from google import genai
from google.genai.types import Content, EmbedContentConfig, Part
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

# Make sure to set the GEMINI_API_KEY environment variable


def embed_content_sync(model_name: str = "gemini-embedding-001") -> str:
    try:
        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        response = client.models.embed_content(model=model_name, contents="Why is the sky blue?")
        print(response.embeddings)

        config = EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        content = Content(
            parts=[
                Part.from_text(text="Why is the sky blue?"),
                Part.from_text(text="What is the capital of France?"),
            ],
        )
        response = client.models.embed_content(
            model=model_name,
            contents=content,
            config=config,
        )

        return response.embeddings or ""

    except Exception as e:
        return f"Error generating response: {str(e)}"


async def embed_content_async(model_name: str = "gemini-embedding-001") -> str:
    try:
        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
        ).aio

        config = EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        content = Content(
            parts=[
                Part.from_text(text="Why is the sky blue?"),
                Part.from_text(text="What is the capital of France?"),
            ],
        )

        async_response = await client.models.embed_content(
            model=model_name, contents=content, config=config
        )
        return async_response.embeddings or ""
    except Exception as e:
        return f"Error generating response: {str(e)}"


if __name__ == "__main__":
    GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    response = embed_content_sync("gemini-embedding-001")
    print(response)

    async_response = asyncio.run(embed_content_async("gemini-embedding-001"))
    print(async_response)
