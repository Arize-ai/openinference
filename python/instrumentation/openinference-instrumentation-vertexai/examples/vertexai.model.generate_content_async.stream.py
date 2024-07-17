import asyncio

import vertexai
from openinference.instrumentation.vertexai import VertexAIInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from vertexai.generative_models import GenerativeModel

endpoint = "http://127.0.0.1:4317"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

VertexAIInstrumentor().instrument(tracer_provider=tracer_provider)

vertexai.init(location="us-central1")
model = GenerativeModel("gemini-1.5-flash")


async def main() -> None:
    response_gen = await model.generate_content_async(
        "Write a haiku.",
        generation_config={"max_output_tokens": 20},
        stream=True,
    )
    async for response in response_gen:
        # print(response.text, end="")
        print(response.candidates[0].content.parts[0]._raw_part.text, end="")


if __name__ == "__main__":
    asyncio.run(main())
