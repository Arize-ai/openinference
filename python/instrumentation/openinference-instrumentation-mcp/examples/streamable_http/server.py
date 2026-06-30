from mcp.server.fastmcp import FastMCP
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.mcp import MCPInstrumentor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer = tracer_provider.get_tracer("mcp-example-server")
MCPInstrumentor().instrument(tracer_provider=tracer_provider)

server = FastMCP(port=8765)


@server.tool()
async def greet(name: str) -> str:
    """Return a greeting. Wrapped in a span so you can see it nested under
    the client's root span in Phoenix."""
    with tracer.start_as_current_span("greet.compute") as span:
        span.set_attribute("greet.name", name)
        return f"Hello, {name}!"


if __name__ == "__main__":
    print(f"Serving MCP streamable-http on http://localhost:{8765}/mcp")
    try:
        server.run(transport="streamable-http")
    finally:
        tracer_provider.shutdown()
