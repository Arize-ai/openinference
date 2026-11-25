"""Check that the AGENT span's output token count matches the result message.

Runs three queries (a plain reply, a resumed session, and one Read tool call),
then compares each AGENT span's llm.token_count.completion with the
output_tokens reported in that query's ResultMessage.usage. Spans go to the
default Phoenix project.

Requirements:
  - Set ANTHROPIC_API_KEY (or be logged in to Claude Code)
  - (Optional) PHOENIX_COLLECTOR_ENDPOINT to point at a remote Phoenix
    Default: http://127.0.0.1:6006/v1/traces (local Phoenix; start it first)
  - (Optional) PHOENIX_API_KEY, sent as a bearer token; required when Phoenix has auth enabled
  - (Optional) CLAUDE_MODEL to pick the model; defaults to the CLI's default model

Usage:
  python examples/output_token_counts.py [run-label]

The run label is stored on each span as metadata.run, so separate runs are
easy to find in Phoenix.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import using_metadata
from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor
from openinference.semconv.trace import SpanAttributes

PHOENIX_ENDPOINT = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces")
PHOENIX_API_KEY = os.environ.get("PHOENIX_API_KEY")
MODEL = os.environ.get("CLAUDE_MODEL")
RUN_LABEL = sys.argv[1] if len(sys.argv) > 1 else "output-token-counts"


def _setup_tracing() -> tuple[trace_sdk.TracerProvider, InMemorySpanExporter]:
    tracer_provider = trace_sdk.TracerProvider()
    headers = {"authorization": f"Bearer {PHOENIX_API_KEY}"} if PHOENIX_API_KEY else None
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(PHOENIX_ENDPOINT, headers=headers))
    )
    memory_exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)
    return tracer_provider, memory_exporter


async def main() -> None:
    tracer_provider, memory_exporter = _setup_tracing()

    # Import after instrument() so the traced query is used.
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query  # noqa: E402

    workdir = Path(tempfile.mkdtemp())
    (workdir / "notes.txt").write_text("The code word is marigold.\n")

    def options(**kwargs: Any) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(model=MODEL, cwd=str(workdir), **kwargs)

    async def run(name: str, prompt: str, opts: ClaudeAgentOptions) -> Optional[str]:
        memory_exporter.clear()
        output_tokens = 0
        session_id: Optional[str] = None
        with using_metadata({"run": RUN_LABEL, "scenario": name}):
            async for message in query(prompt=prompt, options=opts):
                if isinstance(message, ResultMessage):
                    output_tokens += (message.usage or {}).get("output_tokens", 0)
                    session_id = message.session_id
        span = next(
            s for s in memory_exporter.get_finished_spans() if s.name == "ClaudeAgentSDK.query"
        )
        completion = (span.attributes or {}).get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION)
        status = "match" if completion == output_tokens else "MISMATCH"
        print(
            f"{name:<10} result output_tokens={output_tokens:<6} "
            f"span completion={completion!s:<6} {status}  trace_id={span.context.trace_id:032x}"
        )
        return session_id

    session_id = await run("plain", "Write a 100-word story.", options(max_turns=1))
    await run("resume", "Write a 60-word sequel.", options(max_turns=1, resume=session_id))
    await run(
        "read-tool",
        "Read notes.txt, then write a 60-word story that uses its code word.",
        options(max_turns=3, allowed_tools=["Read"]),
    )

    tracer_provider.force_flush()
    print(f"\nDone. View traces in Phoenix at {PHOENIX_ENDPOINT.removesuffix('/v1/traces')}")


if __name__ == "__main__":
    asyncio.run(main())
