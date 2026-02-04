#!/usr/bin/env python3
"""Simple query() usage example with instrumentation.

Install dependencies:
    pip install opentelemetry-sdk opentelemetry-exporter-otlp
    pip install 'arize-phoenix[evals]'
    pip install openinference-instrumentation-claude-code

Run Phoenix in another terminal:
    python -m phoenix.server.main serve

Then run this example:
    python examples/simple_query.py
"""

import anyio
from phoenix.otel import register

from claude_agent_sdk import AssistantMessage, TextBlock, query
from openinference.instrumentation.claude_code import ClaudeCodeInstrumentor

# Configure Phoenix tracer - sends traces to http://localhost:6006
tracer_provider = register(
    project_name="claude-code-demo",  # Project name in Phoenix UI
    endpoint="http://localhost:6006/v1/traces",  # Phoenix endpoint
)

# Instrument Claude Code SDK
ClaudeCodeInstrumentor().instrument(tracer_provider=tracer_provider)


async def main():
    """Run simple query with instrumentation."""
    print("Asking Claude: What is 2 + 2?")
    print("View traces at: http://localhost:6006\n")

    async for message in query(prompt="What is 2 + 2?"):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text}")


if __name__ == "__main__":
    anyio.run(main)
