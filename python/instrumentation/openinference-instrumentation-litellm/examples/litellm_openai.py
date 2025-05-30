import asyncio
import os

import litellm
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.litellm import LiteLLMInstrumentor

if __name__ == "__main__":
    # Set up OpenTelemetry tracing with resource attributes
    endpoint = "http://127.0.0.1:6006/v1/traces"
    resource = Resource.create(
        {
            "service.name": "litellm-openai-example",
            "openinference.project.name": "openinference-litellm-demo",
        }
    )
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

    # Synchronous calls
    litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"content": "What's the capital of China?", "role": "user"}],
    )
    litellm.completion(
        model="gpt-3.5-turbo",
        messages=[
            {"content": "Hello, I want to bake a cake", "role": "user"},
            {"content": "Hello, I can pull up some recipes for cakes.", "role": "assistant"},
            {"content": "No actually I want to make a pie", "role": "user"},
        ],
        temperature=0.7,
    )
    litellm.completion_with_retries(
        model="gpt-3.5-turbo",
        messages=[{"content": "What's the highest grossing film ever", "role": "user"}],
    )
    litellm.embedding(model="text-embedding-ada-002", input=["good morning from litellm"])
    litellm.image_generation(model="dall-e-2", prompt="cute baby otter")

    async def main():
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[
                {"content": "Hello, I want to bake a cake", "role": "user"},
                {"content": "Hello, I can pull up some recipes for cakes.", "role": "assistant"},
                {"content": "No actually I want to make a pie", "role": "user"},
            ],
            temperature=0.7,
            max_tokens=20,
        )
        await litellm.aembedding(
            model="text-embedding-ada-002", input=["good morning from litellm"]
        )
        await litellm.aimage_generation(model="dall-e-2", prompt="cute baby otter")

    asyncio.run(main())
