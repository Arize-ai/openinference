import { randomUUID } from "crypto";
import http from "http";
import { AddressInfo } from "net";

import express from "express";
import { McpServer, ToolCallback } from "@modelcontextprotocol/sdk/server/mcp";
import * as MCPServerSSEModule from "@modelcontextprotocol/sdk/server/sse";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse";
import * as MCPServerStdioModule from "@modelcontextprotocol/sdk/server/stdio";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio";
import * as MCPServerStreamableHTTPModule from "@modelcontextprotocol/sdk/server/streamableHttp";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp";
import { Transport } from "@modelcontextprotocol/sdk/shared/transport";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { Tracer } from "@opentelemetry/api";
import {
  BatchSpanProcessor,
  NodeTracerProvider,
} from "@opentelemetry/sdk-trace-node";
import { z } from "zod";

import { MCPInstrumentation } from "../src";

function newMcpServer(tracer: Tracer) {
  const server = new McpServer({
    name: "MCP",
    version: "0.1.0",
  });

  server.tool("hello", () => {
    return tracer.startActiveSpan("hello", async (span) => {
      const result = await server.server.request(
        {
          method: "whoami",
        },
        z.object({ name: z.string() }),
      );
      try {
        return {
          content: [
            {
              type: "text",
              text: `Hello ${result.name}!`,
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
    serverSSEModule: MCPServerSSEModule,
    serverStdioModule: MCPServerStdioModule,
    serverStreamableHTTPModule: MCPServerStreamableHTTPModule,
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

      await new Promise<void>((resolve, reject) => {
        httpServer = app.listen(0, (err) => {
          if (err) {
            reject(err);
          } else {
            // eslint-disable-next-line no-console
            console.log(
              `Server running on http://localhost:${(httpServer?.address() as AddressInfo).port}/sse`,
            );
            resolve();
          }
        });
      });

      break;
    }
    case "stdio": {
      const server = newMcpServer(tracer);
      transport = new StdioServerTransport();
      await server.connect(transport!);
      break;
    }
    case "streamableHttp": {
      const app = express();
      app.use(express.json());

      const server = newMcpServer(tracer);
      const transports: { [sessionId: string]: StreamableHTTPServerTransport } =
        {};

      app.post("/mcp", async (req, res) => {
        try {
          const sessionId = req.headers["mcp-session-id"] as string | undefined;
          let transport: StreamableHTTPServerTransport;

          if (sessionId && transports[sessionId]) {
            transport = transports[sessionId];
          } else if (!sessionId && isInitializeRequest(req.body)) {
            transport = new StreamableHTTPServerTransport({
              sessionIdGenerator: () => randomUUID(),
              onsessioninitialized: (sessionId: string) => {
                transports[sessionId] = transport;
              },
            });
            await server.connect(transport);
            await transport.handleRequest(req, res, req.body);
            return;
          } else {
            res.status(400).json({
              jsonrpc: "2.0",
              error: {
                code: -32000,
                message: "Bad Request: No valid session ID provided",
              },
              id: null,
            });
            return;
          }

          await transport.handleRequest(req, res, req.body);
        } catch (error) {
          if (!res.headersSent) {
            res.status(500).json({
              jsonrpc: "2.0",
              error: {
                code: -32603,
                message: "Internal server error",
              },
              id: null,
            });
          }
        }
      });

      app.get("/mcp", async (req, res) => {
        const sessionId = req.headers["mcp-session-id"] as string | undefined;
        if (!sessionId || !transports[sessionId]) {
          res.status(400).send("Invalid or missing session ID");
          return;
        }
        const transport = transports[sessionId];
        await transport.handleRequest(req, res);
      });

      await new Promise<void>((resolve, reject) => {
        httpServer = app.listen(0, (err) => {
          if (err) {
            reject(err);
          } else {
            // eslint-disable-next-line no-console
            console.log(
              `Server running on http://localhost:${(httpServer?.address() as AddressInfo).port}/mcp`,
            );
            resolve();
          }
        });
      });

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
