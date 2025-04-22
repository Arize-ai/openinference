# math_server.py
from mcp.server.fastmcp import FastMCP
from phoenix.otel import register

from openinference.instrumentation.mcp import MCPInstrumentor

tracer_provider = register(project_name="langchain-mcp-adapters")
MCPInstrumentor().instrument(tracer_provider=tracer_provider)

tracer = tracer_provider.get_tracer("math-mcp-server")

mcp = FastMCP("Math")


@mcp.tool()
@tracer.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
@tracer.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    mcp.run(transport="stdio")
