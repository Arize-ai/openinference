"""
Shows that ``suppress_tracing`` works both as a synchronous ``with`` and as an
``async with`` context manager. Spans started inside either block are dropped;
spans started after the block are exported again, including after an exception
escapes the block.

Expected traces in Phoenix: exactly five, each a chain span with one tool child, and
none containing "suppressed" in their name:

    sync.traced            -> sync_tool
    sync.after             -> sync_tool
    async.traced           -> async_tool
    async.after            -> async_tool
    async.after_exception  -> async_tool

1. Run a local OTLP collector such as Phoenix: `uvx arize-phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Run this example: `python suppress_tracing_sync_async.py`
4. View the traces at http://localhost:6006 under the
   `instrumentation-suppress-tracing-sync-async` project.
"""

import asyncio
import os

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation import (
    TracerProvider,
    get_span_kind_attributes,
    suppress_tracing,
    tool_span,
)
from openinference.semconv.resource import ResourceAttributes

PHOENIX_BASE_URL = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
COLLECTOR_ENDPOINT = f"{PHOENIX_BASE_URL.rstrip('/')}/v1/traces"
PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "instrumentation-suppress-tracing-sync-async")

tracer_provider = TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: PROJECT_NAME}),
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(COLLECTOR_ENDPOINT)))
tracer = tracer_provider.get_tracer(__name__)


@tool_span(tracer=tracer)
def sync_tool(name: str) -> str:
    return f"sync tool ran: {name}"


@tool_span(tracer=tracer)
async def async_tool(name: str) -> str:
    await asyncio.sleep(0.01)
    return f"async tool ran: {name}"


def run_sync() -> None:
    with tracer.start_as_current_span("sync.traced", attributes=get_span_kind_attributes("chain")):
        sync_tool("traced")

    with suppress_tracing():
        # Neither of these should produce a span.
        with tracer.start_as_current_span(
            "sync.suppressed", attributes=get_span_kind_attributes("chain")
        ):
            sync_tool("suppressed")

    # Suppression must be lifted once the block exits.
    with tracer.start_as_current_span("sync.after", attributes=get_span_kind_attributes("chain")):
        sync_tool("after")


async def run_async() -> None:
    with tracer.start_as_current_span("async.traced", attributes=get_span_kind_attributes("chain")):
        await async_tool("traced")

    async with suppress_tracing():
        # Neither of these should produce a span, even across an await.
        with tracer.start_as_current_span(
            "async.suppressed", attributes=get_span_kind_attributes("chain")
        ):
            await async_tool("suppressed")

    with tracer.start_as_current_span("async.after", attributes=get_span_kind_attributes("chain")):
        await async_tool("after")

    # An exception escaping the block must still restore tracing.
    try:
        async with suppress_tracing():
            await async_tool("suppressed_before_error")
            raise ValueError("boom")
    except ValueError:
        pass

    with tracer.start_as_current_span(
        "async.after_exception", attributes=get_span_kind_attributes("chain")
    ):
        await async_tool("after_exception")


def main() -> None:
    run_sync()
    print("sync: done")
    asyncio.run(run_async())
    print("async: done")
    tracer_provider.force_flush()


if __name__ == "__main__":
    main()
