import asyncio
import sys
import traceback

from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import TokenMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.tool import AnyTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

from examples.setup import setup_observability

setup_observability()

llm = ChatModel.from_name(
    "ollama:granite3.3",
    ChatModelParameters(temperature=0),
)

tools: list[AnyTool] = [
    OpenMeteoTool(),
    DuckDuckGoSearchTool(),
]

agent = ReActAgent(llm=llm, tools=tools, memory=TokenMemory(llm))

prompt = "What's the current weather in Las Vegas?"


async def main() -> None:
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
