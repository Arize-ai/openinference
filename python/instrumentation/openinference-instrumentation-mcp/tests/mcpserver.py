import os
from importlib import import_module
from typing import Any, Literal, cast

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.mcp import MCPInstrumentor

transport = cast(Literal["sse", "stdio", "streamable-http"], os.environ.get("MCP_TRANSPORT"))
port_env = os.environ.get("MCP_PORT")
otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
span_exporter = OTLPSpanExporter(f"{otlp_endpoint}/v1/traces")
tracer_provider = trace_sdk.TracerProvider()
span_processor = SimpleSpanProcessor(span_exporter)
tracer_provider.add_span_processor(span_processor)

tracer = tracer_provider.get_tracer("mcp-test-server")

MCPInstrumentor().instrument(tracer_provider=tracer_provider)

# Make sure instrumentation is loaded before MCP.
from mcp.types import SamplingMessage, TextContent  # noqa: E402

# MCP 2.0 renamed FastMCP to MCPServer and moved its module.
try:
    mcp_server_module = import_module("mcp.server.fastmcp")
    server = mcp_server_module.FastMCP(port=int(port_env) if port_env else 0)
    is_mcp_v2 = False
except ModuleNotFoundError:
    mcp_server_module = import_module("mcp.server.mcpserver")
    server = mcp_server_module.MCPServer()
    is_mcp_v2 = True


async def hello(ctx: Any) -> str:
    with tracer.start_as_current_span("hello"):
        response = await ctx.session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text="What is your name?"),
                )
            ],
            max_tokens=20,
            related_request_id=ctx.request_id,
        )
        assert isinstance(response.content, TextContent)
        return f"Hello {response.content.text}!"


# Both SDK versions identify injected context by its exact runtime annotation.
hello.__annotations__["ctx"] = mcp_server_module.Context
server.tool()(hello)


try:
    if is_mcp_v2 and port_env:
        server.run(transport=transport, port=int(port_env))
    else:
        server.run(transport=transport)
finally:
    tracer_provider.shutdown()
