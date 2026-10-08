import asyncio

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.mcp import MCPInstrumentor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer = tracer_provider.get_tracer("mcp-example-client")
MCPInstrumentor().instrument(tracer_provider=tracer_provider)

from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402


async def main() -> None:
    server_url = "http://localhost:8765/mcp"
    with tracer.start_as_current_span("client.session") as root:
        root.set_attribute("mcp.server.url", server_url)
        async with streamable_http_client(server_url) as (reader, writer, _):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                tools = await session.list_tools()
                print("Available tools:", [t.name for t in tools.tools])

                result = await session.call_tool("greet", {"name": "OpenInference"})
                for block in result.content:
                    if getattr(block, "type", None) == "text":
                        print("Server said:", block.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        tracer_provider.shutdown()
