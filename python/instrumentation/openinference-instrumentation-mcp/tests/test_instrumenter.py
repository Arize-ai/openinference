import asyncio
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import pytest
from mcp import ClientSession
from mcp.shared.session import RequestResponder
from mcp.types import ClientResult, ServerNotification, ServerRequest, TextContent
from opentelemetry.trace import Tracer

from tests.collector import OTLPServer, Telemetry
from tests.whoami import TestClientResult, TestServerRequest, WhoamiResult


# The way MCP SDK creates async tasks means we need this to be called inline with the test,
# not as a fixture.
@asynccontextmanager
async def mcp_client(
    transport: str, tracer: Tracer, otlp_endpoint: str
) -> AsyncGenerator[ClientSession, None]:
    # Lazy import to get instrumented versions. Users will use opentelemetry-instrument or otherwise
    # initialize instrumentation as early as possible and should not run into issues, but we control
    # instrumentation through fixtures instead.
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    async def message_handler(
        message: RequestResponder[ServerRequest, ClientResult] | ServerNotification | Exception,
    ) -> None:
        if not isinstance(message, RequestResponder) or message.request.root.method != "whoami":
            return
        with message as responder, tracer.start_as_current_span("whoami"):
            await responder.respond(TestClientResult(WhoamiResult(name="OpenInference")))  # type: ignore

    server_script = str(Path(__file__).parent / "mcpserver.py")
    pythonpath = str(Path(__file__).parent.parent)
    match transport:
        case "stdio":
            async with stdio_client(
                StdioServerParameters(
                    command=sys.executable,
                    args=[server_script],
                    env={
                        "MCP_TRANSPORT": "stdio",
                        "OTEL_EXPORTER_OTLP_ENDPOINT": otlp_endpoint,
                        "PYTHONPATH": pythonpath,
                    },
                )
            ) as (reader, writer), ClientSession(
                reader, writer, message_handler=message_handler
            ) as client:
                client._receive_request_type = TestServerRequest
                await client.initialize()
                yield client
        case "sse":
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                server_script,
                env={
                    "MCP_TRANSPORT": "sse",
                    "OTEL_EXPORTER_OTLP_ENDPOINT": otlp_endpoint,
                    "PYTHONPATH": pythonpath,
                },
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
                        ), ClientSession(reader, writer, message_handler=message_handler) as client:
                            client._receive_request_type = TestServerRequest
                            await client.initialize()
                            yield client
                        break
            finally:
                proc.kill()
                await proc.wait()
        case "streamable-http":
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                server_script,
                env={
                    "MCP_TRANSPORT": "streamable-http",
                    "OTEL_EXPORTER_OTLP_ENDPOINT": otlp_endpoint,
                    "PYTHONPATH": pythonpath,
                },
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
                        async with streamablehttp_client(f"http://localhost:{port}/mcp") as (
                            reader,
                            writer,
                            _
                        ), ClientSession(reader, writer, message_handler=message_handler) as client:
                            client._receive_request_type = TestServerRequest
                            await client.initialize()
                            yield client
                        break
            finally:
                proc.kill()
                await proc.wait()


@pytest.mark.parametrize("transport", ["sse", "stdio", "streamable-http"])
async def test_hello(
    transport: str, tracer: Tracer, telemetry: Telemetry, otlp_collector: OTLPServer
) -> None:
    async with mcp_client(
        transport, tracer, f"http://localhost:{otlp_collector.server_port}/"
    ) as client:
        with tracer.start_as_current_span("root"):
            tools_res = await client.list_tools()
            assert len(tools_res.tools) == 1
            assert tools_res.tools[0].name == "hello"
            tool_res = await client.call_tool("hello")
            content = tool_res.content[0]
            assert isinstance(content, TextContent)
            assert content.text == "Hello OpenInference!"

    for resource_spans in telemetry.traces:
        for scope_spans in resource_spans.scope_spans:
            match scope_spans.scope.name:
                case "mcp-test-client":
                    for span in scope_spans.spans:
                        match span.name:
                            case "root":
                                root_span = span
                            case "whoami":
                                whoami_span = span
                case "mcp-test-server":
                    server_span = scope_spans.spans[0]
    assert root_span.name == "root"
    assert server_span.name == "hello"
    assert whoami_span.name == "whoami"
    assert server_span.trace_id == root_span.trace_id
    assert server_span.parent_span_id == root_span.span_id
    assert whoami_span.trace_id == root_span.trace_id
    assert whoami_span.parent_span_id == server_span.span_id
