#!/usr/bin/env python3
"""Example using TraceConfig to hide sensitive data.

This demonstrates how to mask inputs/outputs in traces for privacy.

Install dependencies:
    pip install opentelemetry-sdk opentelemetry-exporter-otlp
    pip install 'arize-phoenix[evals]'
    pip install openinference-instrumentation-claude-code

Run Phoenix:
    python -m phoenix.server.main serve

View traces at: http://localhost:6006
"""

import anyio
from phoenix.otel import register

from claude_agent_sdk import AssistantMessage, TextBlock, query
from openinference.instrumentation import TraceConfig
from openinference.instrumentation.claude_code import ClaudeCodeInstrumentor

# Configure Phoenix tracer
tracer_provider = register(
    project_name="claude-code-privacy-demo",
    endpoint="http://localhost:6006/v1/traces",
)

# Configure to hide inputs and outputs
config = TraceConfig(
    hide_inputs=True,
    hide_outputs=True,
)

# Instrument with privacy config
ClaudeCodeInstrumentor().instrument(
    tracer_provider=tracer_provider,
    config=config,
)


async def main():
    """Run query with input/output masking."""
    print("Running query with hidden inputs/outputs...")
    print("View traces at: http://localhost:6006")
    print("Notice: prompts and responses will be masked!\n")

    async for message in query(prompt="What is my secret password?"):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text}")

    print("\nCheck Phoenix UI - inputs/outputs should be masked!")


if __name__ == "__main__":
    anyio.run(main)
