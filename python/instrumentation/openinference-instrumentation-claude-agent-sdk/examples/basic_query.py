"""Basic example: trace Claude Agent SDK query() with OpenInference to Phoenix.

Option A – Phoenix Cloud (no local server):
    Set PHOENIX_COLLECTOR_ENDPOINT and PHOENIX_API_KEY from your Phoenix Cloud space.
    See https://arize.com/docs/phoenix/get-started/get-started-tracing

Option B – Local Phoenix:
    python -m phoenix.server.main serve

Then run this script (set ANTHROPIC_API_KEY):
    python basic_query.py
"""

import asyncio
import os

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor

# Export traces to Phoenix Cloud or local Phoenix (default http://127.0.0.1:6006)
endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces")
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)

# Import after instrument() so the traced query is used (see https://platform.claude.com/docs/en/agent-sdk/overview)
from claude_agent_sdk import query  # noqa: E402


async def main() -> None:
    async for message in query(prompt="What is 2 + 2? Reply in one short sentence."):
        if hasattr(message, "result") and message.result is not None:
            print(message.result)
        elif getattr(message, "subtype", None) == "init":
            print("Session started.")
        else:
            print(message)


if __name__ == "__main__":
    asyncio.run(main())
    print("\nDone. View traces in Phoenix Cloud or at http://127.0.0.1:6006 (local).")
