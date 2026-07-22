from asyncio import run

from agents import Agent, ModelSettings, Runner
from openai.types.shared import Reasoning
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)


reasoning_agent = Agent(
    name="Reasoning Math Tutor",
    instructions="You solve math word problems. "
    "Think step by step before giving the final numeric answer.",
    model="o4-mini",
    model_settings=ModelSettings(
        reasoning=Reasoning(effort="medium", summary="detailed"),
        include=["reasoning.encrypted_content"],
    ),
)


async def main():
    result = await Runner.run(
        reasoning_agent,
        "A train leaves Chicago at 60 mph. Two hours later a second train "
        "leaves the same station on the same track at 90 mph. How long "
        "after the first train departs does the second train catch up?",
    )
    print(result.final_output)


run(main())
