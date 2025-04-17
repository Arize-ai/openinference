import asyncio
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import pytest
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent
from opentelemetry.trace import Tracer

from tests.collector import OTLPServer, Telemetry


# The way MCP SDK creates async tasks means we need this to be called inline with the test,
# not as a fixture.
@asynccontextmanager
async def mcp_client(transport: str, otlp_endpoint: str) -> AsyncGenerator[ClientSession, None]:
    server_script = str(Path(__file__).parent / "mcpserver.py")
    match transport:
        case "stdio":
            async with stdio_client(
                StdioServerParameters(
                    command=sys.executable,
                    args=[server_script],
                    env={"MCP_TRANSPORT": "stdio", "OTEL_EXPORTER_OTLP_ENDPOINT": otlp_endpoint},
                )
            ) as (reader, writer), ClientSession(reader, writer) as client:
                await client.initialize()
                yield client
        case "sse":
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                server_script,
                env={"MCP_TRANSPORT": "sse", "OTEL_EXPORTER_OTLP_ENDPOINT": otlp_endpoint},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                stderr = proc.stderr
                assert stderr is not None
                for i in range(100):
                    line = str(await stderr.readline())
                    if "Uvicorn running on http://0.0.0.0:" in line:
                        _, rest = line.split("http://0.0.0.0:", 1)
                        port, _ = rest.split(" ", 1)
                        async with sse_client(f"http://localhost:{port}/sse") as (
                            reader,
                            writer,
                        ), ClientSession(reader, writer) as client:
                            await client.initialize()
                            yield client
                        break
            finally:
                proc.kill()
                await proc.wait()


@pytest.mark.parametrize("transport", ["sse", "stdio"])
async def test_hello(
    transport: str, tracer: Tracer, telemetry: Telemetry, otlp_collector: OTLPServer
) -> None:
    async with mcp_client(transport, f"http://localhost:{otlp_collector.server_port}/") as client:
        with tracer.start_as_current_span("root"):
            tools_res = await client.list_tools()
            assert len(tools_res.tools) == 1
            assert tools_res.tools[0].name == "hello"
            tool_res = await client.call_tool("hello")
            content = tool_res.content[0]
            assert isinstance(content, TextContent)
            assert content.text == "World!"

    assert len(telemetry.traces) == 2
    for resource_spans in telemetry.traces:
        assert len(resource_spans.scope_spans) == 1
        for scope_spans in resource_spans.scope_spans:
            assert len(scope_spans.spans) == 1
            match scope_spans.scope.name:
                case "mcp-test-client":
                    client_span = scope_spans.spans[0]
                case "mcp-test-server":
                    server_span = scope_spans.spans[0]
    assert client_span.name == "root"
    assert server_span.name == "hello"
    assert server_span.trace_id == client_span.trace_id
    assert server_span.parent_span_id == client_span.span_id
