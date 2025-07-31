import asyncio
import sys
import traceback

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend import UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.tool import AnyTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

from examples.setup import setup_observability

setup_observability()

llm = ChatModel.from_name("ollama:granite3.3:8b", ChatModelParameters(temperature=0.1))

tools: list[AnyTool] = [
    ThinkTool(),
    OpenMeteoTool(),
]


async def main() -> None:
    memory = UnconstrainedMemory()
    await memory.add(UserMessage("My name is Thomas."))

    agent = RequirementAgent(
        llm=llm,
        tools=tools,
        memory=memory,
        requirements=[
            ConditionalRequirement(ThinkTool, force_at_step=1),
            ConditionalRequirement(OpenMeteoTool, force_at_step=2),
        ],
    )

    prompt = "What's the current weather in Las Vegas?"

    response = await agent.run(
        prompt=prompt,
        execution=AgentExecutionConfig(
            max_retries_per_step=3, total_max_retries=10, max_iterations=20
        ),
    )

    print("Agent 🤖 : ", response.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
