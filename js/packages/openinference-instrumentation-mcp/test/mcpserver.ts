import http from "http";
import { AddressInfo } from "net";

import express from "express";
import * as MCPServerModule from "@modelcontextprotocol/sdk/server/index";
import { McpServer, ToolCallback } from "@modelcontextprotocol/sdk/server/mcp";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio";
import { Transport } from "@modelcontextprotocol/sdk/shared/transport";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { Tracer } from "@opentelemetry/api";
import {
  BatchSpanProcessor,
  NodeTracerProvider,
} from "@opentelemetry/sdk-trace-node";

import { MCPInstrumentation } from "../src";

function newMcpServer(tracer: Tracer) {
  const server = new McpServer({
    name: "MCP",
    version: "0.1.0",
  });

  server.tool("hello", () => {
    return tracer.startActiveSpan("hello", (span) => {
      try {
        return {
          content: [
            {
              type: "text",
              text: "World!",
            },
          ],
        };
      } finally {
        span.end();
      }
    }) as ReturnType<ToolCallback>;
  });
  return server;
}

async function main() {
  const transportType = process.env.MCP_TRANSPORT!;

  const otlpEndpoint = process.env.OTEL_EXPORTER_OTLP_ENDPOINT!;
  const exporter = new OTLPTraceExporter({
    url: `${otlpEndpoint}/v1/traces`,
  });
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new BatchSpanProcessor(exporter)],
  });
  tracerProvider.register();
  const tracer = tracerProvider.getTracer("mcp-test-server");

  const instrumentation = new MCPInstrumentation();
  instrumentation.setTracerProvider(tracerProvider);
  instrumentation.manuallyInstrument({
    serverModule: MCPServerModule,
  });

  let transport: Transport;
  let httpServer: http.Server | undefined;
  switch (transportType) {
    case "sse": {
      const app = express();

      let servers: McpServer[] = [];

      app.get("/sse", async (req, res) => {
        const server = newMcpServer(tracer);
        servers.push(server);

        const transport = new SSEServerTransport("/message", res);

        server.server.onclose = () => {
          servers = servers.filter((s) => s !== server);
        };

        await server.connect(transport);
      });

      app.post("/message", async (req, res) => {
        const sessionId = req.query.sessionId;
        const transport = servers
          .map((s) => s.server.transport)
          .find((t) => t!.sessionId === sessionId) as
          | SSEServerTransport
          | undefined;
        if (!transport) {
          res.status(404).send("Session not found");
          return;
        }

        await transport.handlePostMessage(req, res);
      });

      const { promise, resolve } = Promise.withResolvers();
      httpServer = app.listen(0, () => {
        // eslint-disable-next-line no-console
        console.log(
          `Server running on http://localhost:${(httpServer?.address() as AddressInfo).port}/sse`,
        );
        resolve(httpServer);
      });
      await promise;

      break;
    }
    case "stdio": {
      const server = newMcpServer(tracer);
      transport = new StdioServerTransport();
      await server.connect(transport!);
      break;
    }
  }

  process.on("SIGTERM", async () => {
    await tracerProvider.shutdown();
    if (httpServer) {
      httpServer.close();
    }
    process.exit();
  });
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});
