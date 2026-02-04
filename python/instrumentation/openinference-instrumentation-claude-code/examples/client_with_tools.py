#!/usr/bin/env python3
"""ClaudeSDKClient with tools example.

This example shows Claude using tools (Read, Write) with full tracing.

Install dependencies:
    pip install opentelemetry-sdk opentelemetry-exporter-otlp
    pip install 'arize-phoenix[evals]'
    pip install openinference-instrumentation-claude-code
    pip install claude-agent-sdk

Authentication:
    Option 1: Use Claude Code CLI authentication (run 'claude' once to authenticate)
    Option 2: Set environment variable: export ANTHROPIC_API_KEY='your-api-key'

    Get your API key from: https://console.anthropic.com/

Run Phoenix:
    python -m phoenix.server.main serve

View traces at: http://localhost:6006
"""

import anyio
from phoenix.otel import register

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, TextBlock
from openinference.instrumentation.claude_code import ClaudeCodeInstrumentor

# Configure Phoenix tracer
tracer_provider = register(
    project_name="claude-code-tools-demo",
    endpoint="http://localhost:6006/v1/traces",
)

# Instrument Claude Code SDK
ClaudeCodeInstrumentor().instrument(tracer_provider=tracer_provider)


async def main():
    """Run client with tools."""
    print("Running Claude with Read/Write tools...")
    print("View traces at: http://localhost:6006\n")

    # Configure options with tools
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write"],
        system_prompt="You are a helpful file assistant.",
    )

    # Use client
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Create a file called hello.txt with 'Hello, World!'")

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")


if __name__ == "__main__":
    anyio.run(main)
