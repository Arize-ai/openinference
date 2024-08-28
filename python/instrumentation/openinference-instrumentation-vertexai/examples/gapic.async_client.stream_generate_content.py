import asyncio
import os

from google.cloud.aiplatform import initializer
from google.cloud.aiplatform_v1 import (
    Content,
    GenerateContentRequest,
    GenerationConfig,
    Part,
    PredictionServiceAsyncClient,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.vertexai import VertexAIInstrumentor

endpoint = "http://127.0.0.1:4317"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

VertexAIInstrumentor().instrument(tracer_provider=tracer_provider)

location = "us-central1"
project = os.environ["CLOUD_ML_PROJECT_ID"]
model = "gemini-1.5-flash"
request = GenerateContentRequest(
    {
        "contents": [Content(dict(role="user", parts=[Part(dict(text="Write a haiku."))]))],
        "model": f"projects/{project}/locations/{location}/publishers/google/models/{model}",
        "generation_config": GenerationConfig(dict(max_output_tokens=20)),
    }
)


async def main() -> None:
    client: PredictionServiceAsyncClient = initializer.global_config.create_client(
        client_class=PredictionServiceAsyncClient,
        location_override=location,
        prediction_client=True,
    )
    response_gen = await client.stream_generate_content(request)
    async for response in response_gen:
        print(response.candidates[0].content.parts[0].text, end="")


if __name__ == "__main__":
    asyncio.run(main())
