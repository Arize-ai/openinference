"""The same Assistant against an OpenAI-compatible server.

Qwen-Agent's `oai` backend discards the model response's usage block, so this
example also enables OpenAIInstrumentor: the nested OpenAI-SDK span carries the
token counts, and the Qwen-Agent LLM span deliberately does not, so trace-level
totals are not double-counted.

Defaults to a local Ollama serving Qwen — `ollama pull qwen3`. Point
QWEN_MODEL_SERVER at a vLLM server or at DashScope's compatible-mode endpoint
(https://dashscope.aliyuncs.com/compatible-mode/v1) to use those instead.
"""

import json
import os
from typing import Any

from phoenix.otel import register
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.qwen_agent import QwenAgentInstrumentor

tracer_provider = register(project_name="qwen-agent-openai-compatible", auto_instrument=False)
QwenAgentInstrumentor().instrument(tracer_provider=tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


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
        return json.dumps({"city": args["city"], "temperature_c": 21, "sky": "sunny"})


def main() -> None:
    bot = Assistant(
        llm={
            "model": os.environ.get("QWEN_MODEL", "qwen3"),
            "model_type": "oai",
            "model_server": os.environ.get("QWEN_MODEL_SERVER", "http://localhost:11434/v1"),
            "api_key": os.environ.get("QWEN_API_KEY", "EMPTY"),
        },
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
