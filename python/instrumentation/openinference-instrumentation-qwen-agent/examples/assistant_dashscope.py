"""An Assistant with a tool, traced on the DashScope backend.

DashScope exposes token usage to Qwen-Agent, so the LLM span carries
`llm.token_count.*` without any additional instrumentor.

Requires DASHSCOPE_API_KEY.
"""

import json
import os
from typing import Any

from phoenix.otel import register
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

from openinference.instrumentation.qwen_agent import QwenAgentInstrumentor

tracer_provider = register(project_name="qwen-agent-dashscope", auto_instrument=False)
QwenAgentInstrumentor().instrument(tracer_provider=tracer_provider)


@register_tool("get_weather")
class WeatherTool(BaseTool):
    description = "Get the current weather for a city."
    parameters = {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    }

    def call(self, params: Any, **kwargs: Any) -> str:
        args = self._verify_json_format_args(params)
        # A real tool would call a weather API here.
        return json.dumps({"city": args["city"], "temperature_c": 21, "sky": "sunny"})


def main() -> None:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is required for the DashScope backend")

    bot = Assistant(
        llm={"model": os.environ.get("QWEN_MODEL", "qwen-max"), "model_type": "qwen_dashscope"},
        name="weather-assistant",
        description="Answers questions about the weather.",
        system_message="You are a concise, helpful assistant.",
        function_list=["get_weather"],
    )

    messages = [{"role": "user", "content": "What is the weather in Beijing?"}]
    for responses in bot.run(messages):
        pass
    print(responses[-1]["content"])


if __name__ == "__main__":
    main()
