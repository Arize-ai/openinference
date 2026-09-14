"""A Router delegating to two specialised assistants.

Router and GroupChat call `run` on their member agents, so nested agents show up
as child AGENT spans with no extra configuration.

Requires DASHSCOPE_API_KEY.
"""

import os

from phoenix.otel import register
from qwen_agent.agents import Assistant, Router

from openinference.instrumentation.qwen_agent import QwenAgentInstrumentor

tracer_provider = register(project_name="qwen-agent-router", auto_instrument=False)
QwenAgentInstrumentor().instrument(tracer_provider=tracer_provider)


def main() -> None:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is required for the DashScope backend")

    llm_cfg = {"model": os.environ.get("QWEN_MODEL", "qwen-max"), "model_type": "qwen_dashscope"}

    poet = Assistant(
        llm=llm_cfg,
        name="poet",
        description="Writes short poems.",
        system_message="You write four-line poems and nothing else.",
    )
    mathematician = Assistant(
        llm=llm_cfg,
        name="mathematician",
        description="Answers arithmetic questions.",
        system_message="You answer arithmetic questions with just the number.",
    )
    router = Router(llm=llm_cfg, agents=[poet, mathematician])

    for responses in router.run([{"role": "user", "content": "Write a poem about tracing."}]):
        pass
    print(responses[-1]["content"])


if __name__ == "__main__":
    main()
