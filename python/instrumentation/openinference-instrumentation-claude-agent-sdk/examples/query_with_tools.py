"""Example: trace a Claude Agent SDK query() that uses tools (e.g. Bash, Glob).

Each query() run is traced as one CHAIN span. For LLM/tool spans inside the run,
also instrument Anthropic (openinference-instrumentation-anthropic).

Use Phoenix Cloud (https://arize.com/docs/phoenix/get-started/get-started-tracing)
or run local Phoenix: python -m phoenix.server.main serve

Then run (set ANTHROPIC_API_KEY):
    python query_with_tools.py
"""

import asyncio

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)

# Import after instrument() so the traced query is used (see https://platform.claude.com/docs/en/agent-sdk/overview)
from claude_agent_sdk import ClaudeAgentOptions, query  # noqa: E402


async def main() -> None:
    options = ClaudeAgentOptions(
        allowed_tools=["Bash", "Glob"],
        permission_mode="bypassPermissions",
    )
    async for message in query(
        prompt="List files in the current directory and say how many there are.",
        options=options,
    ):
        if hasattr(message, "result") and message.result is not None:
            print(message.result)
        elif getattr(message, "subtype", None) == "init":
            print("Session started.")


if __name__ == "__main__":
    asyncio.run(main())
    print("\nDone. View traces at http://127.0.0.1:6006")
