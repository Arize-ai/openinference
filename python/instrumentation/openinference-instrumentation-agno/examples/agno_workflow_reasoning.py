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

math_tutor = Agent(
    name="Math Tutor",
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
    name="Tutoring Team",
    model=OpenAIResponses(
        id="o4-mini",
        reasoning={
            "effort": "high",
            "summary": "detailed",
        },
    ),
    members=[math_tutor],
)

workflow = Workflow(
    name="Math Tutoring Workflow",
    steps=[team],
)


async def main():
    result = await workflow.arun(
        input=(
            "A farmer has 17 sheep, and all but 9 die. How many sheep are left? "
            "Then, if the remaining sheep are split evenly into 3 pens, "
            "how many sheep are in each pen?"
        ),
        stream=False,
    )
    print(result)


asyncio.run(main())
