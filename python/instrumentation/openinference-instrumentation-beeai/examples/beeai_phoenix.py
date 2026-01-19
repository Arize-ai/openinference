import asyncio

from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import (
    ConditionalRequirement,
)
from beeai_framework.errors import FrameworkError
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.handoff import HandoffTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool
from dotenv import load_dotenv
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.beeai import BeeAIInstrumentor

load_dotenv()

endpoint = "http://localhost:6006/v1/traces"

tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

BeeAIInstrumentor().instrument(tracer_provider=tracer_provider)


async def main() -> None:
    knowledge_agent = RequirementAgent(
        llm=OpenAIChatModel(model_id="gpt-4o-mini"),
        # llm=ChatModel.from_name("ollama:granite3.3:8b"),
        tools=[ThinkTool(), WikipediaTool()],
        requirements=[ConditionalRequirement(ThinkTool, force_at_step=1)],
        role="Knowledge Specialist",
        instructions="Provide answers to general questions about the world.",
    )

    weather_agent = RequirementAgent(
        llm=OpenAIChatModel(model_id="gpt-4o-mini"),
        tools=[OpenMeteoTool()],
        role="Weather Specialist",
        instructions="Provide weather forecast for a given destination.",
    )

    main_agent = RequirementAgent(
        name="MainAgent",
        llm=OpenAIChatModel(model_id="gpt-4o-mini"),
        # llm=ChatModel.from_name("ollama:granite3.3:8b"),
        tools=[
            ThinkTool(),
            HandoffTool(
                knowledge_agent,
                name="KnowledgeLookup",
                description="Consult the Knowledge Agent for general questions.",
            ),
            HandoffTool(
                weather_agent,
                name="WeatherLookup",
                description="Consult the Weather Agent for forecasts.",
            ),
        ],
        requirements=[ConditionalRequirement(ThinkTool, force_at_step=1)],
        # Log all tool calls to the console for easier debugging
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
    )

    question = (
        "If I travel to Rome next weekend, what should I expect in terms of weather, "
        "and also tell me one famous historical landmark there?"
    )
    print(f"User: {question}")

    try:
        response = await main_agent.run(question, expected_output="Helpful and clear response.")
        print("Agent:", response)
    except FrameworkError as err:
        print("Error:", err.explain())


if __name__ == "__main__":
    asyncio.run(main())
