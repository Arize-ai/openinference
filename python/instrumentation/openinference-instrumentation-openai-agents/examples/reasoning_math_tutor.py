from asyncio import run

from agents import Agent, ModelSettings, Runner
from openai.types.shared import Reasoning
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from openinference.semconv.resource import ResourceAttributes

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider(
    resource=Resource.create({ResourceAttributes.PROJECT_NAME: "openai-agents-reasoning"})
)
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
    first_result = await Runner.run(
        reasoning_agent,
        "A train leaves Chicago at 60 mph. Two hours later a second train "
        "leaves the same station on the same track at 90 mph. How long "
        "after the first train departs does the second train catch up?",
    )
    print(first_result.final_output)

    # Replay the first turn, including its reasoning item, into a continuation.
    # The second LLM span should contain that item under llm.input_messages.
    second_result = await Runner.run(
        reasoning_agent,
        first_result.to_input_list()
        + [{"role": "user", "content": "Restate the answer in minutes."}],
    )
    print(second_result.final_output)


run(main())
