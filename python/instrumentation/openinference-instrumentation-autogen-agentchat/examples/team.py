import asyncio

import vcr
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
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

# Create a VCR instance
my_vcr = vcr.VCR(
    cassette_library_dir="cassettes",
    record_mode="once",
)


async def main() -> None:
    model_client = OpenAIChatCompletionClient(
        model="gpt-4",
    )

    # Create two agents: a primary and a critic
    primary_agent = AssistantAgent(
        "primary",
        model_client=model_client,
        system_message="You are a helpful AI assistant.",
    )

    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message="""
        Provide constructive feedback.
        Respond with 'APPROVE' when your feedbacks are addressed.
        """,
    )

    # Termination condition: stop when the critic says "APPROVE"
    text_termination = TextMentionTermination("APPROVE")

    # Create a team with both agents
    team = RoundRobinGroupChat(
        [primary_agent, critic_agent], termination_condition=text_termination
    )

    # Run the team on a task
    with my_vcr.use_cassette("team.yaml"):
        result = await team.run(task="Write a short poem about the fall season.")
        await model_client.close()
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
