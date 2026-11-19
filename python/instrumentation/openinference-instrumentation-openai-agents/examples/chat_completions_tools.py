import asyncio
import os

from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool
from openai import AsyncOpenAI
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from openinference.semconv.resource import ResourceAttributes

model_name = os.environ.get("TOOLS_MODEL", "gpt-4o-mini")
project_name = os.environ.get("PHOENIX_PROJECT", "default")
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider(resource=Resource({ResourceAttributes.PROJECT_NAME: project_name}))
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)


@function_tool
def add(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b


async def main() -> None:
    chat_completions_agent = Agent(
        name="Calculator",
        instructions="Use the add tool to answer arithmetic questions.",
        model=OpenAIChatCompletionsModel(model=model_name, openai_client=AsyncOpenAI()),
        tools=[add],
    )
    responses_agent = chat_completions_agent.clone(model=model_name)

    for workflow_name, agent in [
        ("chat-completions-tools", chat_completions_agent),
        ("responses-tools", responses_agent),
    ]:
        result = await Runner.run(
            agent,
            "What is 2 + 2?",
            run_config=RunConfig(workflow_name=workflow_name),
        )
        print(f"{workflow_name}: {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
