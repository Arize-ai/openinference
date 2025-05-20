import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.autogen_agentchat import AutogenAgentChatInstrumentor

# Set up the tracer provider with both OTLP and Console exporters
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

# Instrument the AutogenAgentChat
AutogenAgentChatInstrumentor().instrument(tracer_provider=tracer_provider)


async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    agent1 = AssistantAgent("Assistant1", model_client=model_client)
    agent2 = AssistantAgent("Assistant2", model_client=model_client)
    termination = MaxMessageTermination(3)
    team = RoundRobinGroupChat([agent1, agent2], termination_condition=termination)

    stream = team.run_stream(task="Count from 1 to 10, respond one at a time.")
    async for message in stream:
        print(message)


if __name__ == "__main__":
    asyncio.run(main())
