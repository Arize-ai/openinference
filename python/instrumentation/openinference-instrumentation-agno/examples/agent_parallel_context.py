"""
This example shows how to instrument your agno agent with OpenInference
and send traces to Arize Phoenix

Install dependencies:
pip install openai opentelemetry-sdk opentelemetry-exporter-otlp
pip install openinference-instrumentation-agno
"""
import asyncio

from agno.agent.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team import Team
from agno.workflow.workflow import Workflow

from openinference.instrumentation.agno import AgnoInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

AgnoInstrumentor().instrument(tracer_provider=tracer_provider)

agent_a = Agent(
    name="Agent A",
    model=OpenAIChat(id="gpt-4o-mini"),
    role="You answer questions briefly.",
)

team = Team(
    name="My Team",
    model=OpenAIChat(id="gpt-4o-mini"),
    members=[agent_a],
)

workflow = Workflow(
    name="My Workflow",
    steps=[team],
)


async def main():
    stream = workflow.arun(input="Say hello", stream=True, stream_events=True)
    async for event in stream:
        print(event)  # consume the stream


asyncio.run(main())
