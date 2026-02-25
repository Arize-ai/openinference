"""Example: trace ClaudeSDKClient (one query + receive_response turn).

Uses ClaudeSDKClient for a single exchange. Each receive_response() iteration
is traced as one AGENT span.

Option A – Phoenix Cloud: set PHOENIX_COLLECTOR_ENDPOINT (and PHOENIX_API_KEY).
Option B – Local Phoenix: python -m phoenix.server.main serve

Then run (set ANTHROPIC_API_KEY):
    python client_basic.py
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

# Import after instrument() so the client is traced
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient  # noqa: E402
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock  # noqa: E402


async def main() -> None:
    options = ClaudeAgentOptions(allowed_tools=["Bash", "Glob"])
    async with ClaudeSDKClient(options=options) as client:
        await client.query("What is 2 + 2? Reply in one short sentence.")
        reply: list[str] = []
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        reply.append(block.text)
            elif isinstance(message, ResultMessage) and message.result and not reply:
                reply.append(message.result)
        print("".join(reply) if reply else "")

    print("Done. View traces at http://127.0.0.1:6006 (local) or Phoenix Cloud.")


if __name__ == "__main__":
    asyncio.run(main())
