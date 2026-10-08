"""
This example shows how to instrument your agno agent with OpenInference
and send traces to Arize Phoenix.

Install dependencies:
pip install openai opentelemetry-sdk opentelemetry-exporter-otlp
pip install openinference-instrumentation-agno
"""

import asyncio

from agno.agent.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.workflow.workflow import Workflow
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.agno import AgnoInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

AgnoInstrumentor().instrument(tracer_provider=tracer_provider)

QUESTION = (
    "A farmer has 17 sheep, and all but 9 die. How many sheep are left? "
    "Then, if the remaining sheep are split evenly into 3 pens, "
    "how many sheep are in each pen?"
)


def _build_workflow(name_suffix: str) -> Workflow:
    # Fresh agent/team/workflow instances per run so spans from the two
    # scenarios don't share state.
    math_tutor = Agent(
        name=f"Math Tutor {name_suffix}",
        model=OpenAIResponses(
            id="o4-mini",
            reasoning={
                "effort": "high",
                "summary": "detailed",
            },
        ),
        role=(
            "You are a math tutor. Work through problems step by step using "
            "internal reasoning, then give a clear final answer."
        ),
    )

    team = Team(
        name=f"Tutoring Team {name_suffix}",
        model=OpenAIResponses(
            id="o4-mini",
            reasoning={
                "effort": "high",
                "summary": "detailed",
            },
        ),
        members=[math_tutor],
    )

    return Workflow(
        name=f"Math Tutoring Workflow {name_suffix}",
        steps=[team],
    )


async def run_non_streaming() -> None:
    workflow = _build_workflow("(non-streaming)")
    result = await workflow.arun(input=QUESTION, stream=False)
    print(result)


async def run_streaming() -> None:
    workflow = _build_workflow("(streaming)")
    stream = workflow.arun(input=QUESTION, stream=True)
    async for event in stream:
        print(event)  # consume the stream so spans are finalized


async def main() -> None:
    await run_non_streaming()
    await run_streaming()


asyncio.run(main())
