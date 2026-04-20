"""Tool-discovery client over streamable-http, traced end-to-end.

Complements `client.py` (which focuses on `call_tool`). This script exercises
the `tools/list` code path: each list_tools RPC is wrapped in a user span so
the discovery phase is itself visible in Phoenix, and the MCP instrumentation
injects the active trace context into the request's `_meta` — so even though
`tools/list` doesn't hit a user tool handler on the server, any server-side
telemetry triggered by it lands on the same trace.

Run `server.py` first, then this script.
"""

import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.mcp import MCPInstrumentor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer = tracer_provider.get_tracer("mcp-example-list-tools")
MCPInstrumentor().instrument(tracer_provider=tracer_provider)


async def discover_and_invoke(server_url: str) -> None:
    with tracer.start_as_current_span("discovery") as root:
        root.set_attribute("mcp.server.url", server_url)
        async with streamable_http_client(server_url) as (reader, writer, _):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

                with tracer.start_as_current_span("mcp.tools.list") as list_span:
                    tools_result = await session.list_tools()
                    tool_names = [t.name for t in tools_result.tools]
                    list_span.set_attribute("mcp.tools.count", len(tool_names))
                    list_span.set_attribute("mcp.tools.names", tool_names)

                print(f"Discovered {len(tool_names)} tool(s): {tool_names}")

                # Invoke every discovered tool with a stub arg so the full
                # discover → call pipeline is exercised under one trace.
                for name in tool_names:
                    with tracer.start_as_current_span(f"mcp.tools.call:{name}"):
                        result = await session.call_tool(name, {"name": "OpenInference"})
                        for block in result.content:
                            if getattr(block, "type", None) == "text":
                                print(f"  {name} -> {block.text}")


if __name__ == "__main__":
    try:
        asyncio.run(discover_and_invoke("http://localhost:8765/mcp"))
    finally:
        tracer_provider.shutdown()
