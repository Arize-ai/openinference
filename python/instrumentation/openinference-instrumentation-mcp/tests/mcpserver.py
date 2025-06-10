import os
from typing import Literal, cast

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.mcp import MCPInstrumentor

transport = cast(Literal["sse", "stdio", "streamable-http"], os.environ.get("MCP_TRANSPORT"))
otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
span_exporter = OTLPSpanExporter(f"{otlp_endpoint}/v1/traces")
tracer_provider = trace_sdk.TracerProvider()
span_processor = SimpleSpanProcessor(span_exporter)
tracer_provider.add_span_processor(span_processor)

tracer = tracer_provider.get_tracer("mcp-test-server")

MCPInstrumentor().instrument(tracer_provider=tracer_provider)

# Make sure instrumentation is loaded before MCP.
from mcp.server.fastmcp import Context, FastMCP  # noqa: E402

from tests.whoami import TestClientResult, WhoamiRequest  # noqa: E402

server = FastMCP(port=0)


@server.tool()
async def hello(ctx: Context) -> str:  # type: ignore
    with tracer.start_as_current_span("hello"):
        response = await ctx.session.send_request(WhoamiRequest(method="whoami"), TestClientResult)
        name = response.root.name
        return f"Hello {name}!"


try:
    server.run(transport=transport)
finally:
    tracer_provider.shutdown()
