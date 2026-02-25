"""Example: trace ClaudeSDKClient with multiple turns (continuous conversation).

Each query() + receive_response() turn is traced as a separate AGENT span,
so you get one span per user message in the conversation.

Option A – Phoenix Cloud: set PHOENIX_COLLECTOR_ENDPOINT (and PHOENIX_API_KEY).
Option B – Local Phoenix: python -m phoenix.server.main serve

Then run (set ANTHROPIC_API_KEY):
    python client_multi_turn.py
"""

import asyncio
import os

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor

endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces")
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)

# Import after instrument()
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient  # noqa: E402
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock  # noqa: E402


async def main() -> None:
    options = ClaudeAgentOptions(allowed_tools=["Bash", "Glob"])
    async with ClaudeSDKClient(options=options) as client:
        # Turn 1
        await client.query("What is the capital of France? One word.")
        print("Turn 1 – You: What is the capital of France? One word.")
        reply1: list[str] = []
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        reply1.append(block.text)
            elif isinstance(message, ResultMessage) and message.result and not reply1:
                reply1.append(message.result)
        print("Turn 1 – Claude:", "".join(reply1) if reply1 else "")
        print()

        # Turn 2 – same session, Claude remembers context
        await client.query("What is its population? One short sentence.")
        print("Turn 2 – You: What is its population? One short sentence.")
        reply2: list[str] = []
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        reply2.append(block.text)
            elif isinstance(message, ResultMessage) and message.result and not reply2:
                reply2.append(message.result)
        print("Turn 2 – Claude:", "".join(reply2) if reply2 else "")

    print("\nDone. View traces at http://127.0.0.1:6006 (local) or Phoenix Cloud.")


if __name__ == "__main__":
    asyncio.run(main())
