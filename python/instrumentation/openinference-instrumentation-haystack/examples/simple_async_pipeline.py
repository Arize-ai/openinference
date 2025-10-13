import asyncio

from haystack import AsyncPipeline, Pipeline
from haystack.components.generators import OpenAIGenerator
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.haystack import HaystackInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
trace_api.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

# Instrument the Haystack application
HaystackInstrumentor().instrument()


def run():
    pipeline = Pipeline()
    llm = OpenAIGenerator(model="gpt-3.5-turbo")
    pipeline.add_component("llm", llm)
    question = "What is the location of the Hanging Gardens of Babylon?"
    response = pipeline.run({"llm": {"prompt": question}})
    print(response)


async def run_async():
    pipeline = AsyncPipeline()
    llm = OpenAIGenerator(model="gpt-3.5-turbo")
    pipeline.add_component("llm", llm)
    question = "What is the location of the Hanging Gardens of Babylon?"
    response = await pipeline.run_async({"llm": {"prompt": question}})
    print(response)


if __name__ == "__main__":
    # run()
    asyncio.run(run_async())
