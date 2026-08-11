"""Single example: trace Claude Agent SDK and print captured span attributes.

Requirements:
  - Set ANTHROPIC_API_KEY
  - (Optional) PHOENIX_COLLECTOR_ENDPOINT for Phoenix Cloud
    Default: http://127.0.0.1:6006/v1/traces (local Phoenix)
"""

import asyncio
import json
import os
from typing import Optional

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor

PHOENIX_ENDPOINT = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces")

_TASK_PROMPT = (
    "Use the Task tool to delegate a sub-agent. The sub-agent must use the Bash tool "
    "to run: `printf 'hello' | wc -c`. Return exactly the numeric output and nothing else."
)


def _setup_tracing() -> tuple[trace_sdk.TracerProvider, InMemorySpanExporter]:
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(PHOENIX_ENDPOINT)))
    memory_exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)
    return tracer_provider, memory_exporter


async def main() -> None:
    tracer_provider, memory_exporter = _setup_tracing()

    # Import after instrument() so the traced query is used.
    from claude_agent_sdk import ClaudeAgentOptions, query  # noqa: E402

    options = ClaudeAgentOptions(
        allowed_tools=["Bash", "Task", "TaskOutput"],
        permission_mode="bypassPermissions",
    )

    result: Optional[str] = None
    async for message in query(prompt=_TASK_PROMPT, options=options):
        if getattr(message, "result", None) is not None:
            result = message.result

    print(f"Result: {result!r}")

    tracer_provider.force_flush()
    spans = memory_exporter.get_finished_spans()
    print("\nCaptured spans and attributes:")
    for span in spans:
        print(f"- {span.name}")
        print(json.dumps(dict(span.attributes or {}), indent=2, sort_keys=True, default=str))

    print("\nDone. View traces in Phoenix Cloud or at http://127.0.0.1:6006 (local).")


if __name__ == "__main__":
    asyncio.run(main())
