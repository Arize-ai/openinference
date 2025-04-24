import { spawn } from "child_process";
import http from "http";
import { AddressInfo } from "net";
import path from "path";

import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import * as MCPClientModule from "@modelcontextprotocol/sdk/client/index";
import * as MCPClientSSEModule from "@modelcontextprotocol/sdk/client/sse";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse";
import * as MCPClientStreamableHTTPModule from "@modelcontextprotocol/sdk/client/streamableHttp";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp";
import * as MCPClientStdioModule from "@modelcontextprotocol/sdk/client/stdio";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio";
import * as MCPServerStdioModule from "@modelcontextprotocol/sdk/server/stdio";
import * as MCPServerSSEModule from "@modelcontextprotocol/sdk/server/sse";
import * as MCPServerStreamableHTTPModule from "@modelcontextprotocol/sdk/server/streamableHttp";
import { Tracer } from "@opentelemetry/api";
import { z } from "zod";

import { ExportedSpan, startCollector, Telemetry } from "./collector";
import { isPatched, MCPInstrumentation } from "../src";

describe("MCPInstrumentation", () => {
  const telemetry = new Telemetry();

  let collector: http.Server;
  let tracerProvider: NodeTracerProvider;
  let tracer: Tracer;
  let otlpEndpoint: string;

  const instrumentation = new MCPInstrumentation();
  instrumentation.disable();

  beforeAll(async () => {
    collector = await startCollector(telemetry);
    otlpEndpoint = `http://localhost:${(collector.address() as AddressInfo).port}`;
    const exporter = new OTLPTraceExporter({
      url: `${otlpEndpoint}/v1/traces`,
    });
    tracerProvider = new NodeTracerProvider({
      spanProcessors: [new BatchSpanProcessor(exporter)],
    });
    tracerProvider.register();
    tracer = tracerProvider.getTracer("mcp-test-client");

    instrumentation.setTracerProvider(tracerProvider);
    instrumentation.enable();

    instrumentation.manuallyInstrument({
      clientSSEModule: MCPClientSSEModule,
      serverSSEModule: MCPServerSSEModule,
      clientStdioModule: MCPClientStdioModule,
      serverStdioModule: MCPServerStdioModule,
      clientStreamableHTTPModule: MCPClientStreamableHTTPModule,
      serverStreamableHTTPModule: MCPServerStreamableHTTPModule,
    });
  });

  afterAll(async () => {
    instrumentation.disable();
    await tracerProvider.shutdown();

    await new Promise<void>((resolve, reject) => {
      collector.close((err) => {
        if (err) {
          reject(err);
        } else {
          resolve();
        }
      });
    });
  });

  beforeEach(() => {
    telemetry.clear();
  });
  it("is patched", () => {
    expect(
      (MCPClientSSEModule as { openInferencePatched?: boolean })
        .openInferencePatched,
    ).toBe(true);
    expect(isPatched("@modelcontextprotocol/sdk/client/sse")).toBe(true);
    expect(
      (MCPServerSSEModule as { openInferencePatched?: boolean })
        .openInferencePatched,
    ).toBe(true);
    expect(isPatched("@modelcontextprotocol/sdk/server/sse")).toBe(true);
    expect(
      (MCPClientStdioModule as { openInferencePatched?: boolean })
        .openInferencePatched,
    ).toBe(true);
    expect(isPatched("@modelcontextprotocol/sdk/client/stdio")).toBe(true);
    expect(
      (MCPServerStdioModule as { openInferencePatched?: boolean })
        .openInferencePatched,
    ).toBe(true);
    expect(isPatched("@modelcontextprotocol/sdk/server/stdio")).toBe(true);
    expect(
      (MCPClientStreamableHTTPModule as { openInferencePatched?: boolean })
        .openInferencePatched,
    ).toBe(true);
    expect(isPatched("@modelcontextprotocol/sdk/client/streamableHttp")).toBe(
      true,
    );
    expect(
      (MCPServerStreamableHTTPModule as { openInferencePatched?: boolean })
        .openInferencePatched,
    ).toBe(true);
    expect(isPatched("@modelcontextprotocol/sdk/server/streamableHttp")).toBe(
      true,
    );
  });
  it("propagates context - stdio", async () => {
    const transport = new StdioClientTransport({
      command: "tsx",
      args: [path.join(__dirname, "mcpserver.ts")],
      env: {
        ...process.env,
        MCP_TRANSPORT: "stdio",
        OTEL_EXPORTER_OTLP_ENDPOINT: otlpEndpoint,
      },
    });
    const client = new MCPClientModule.Client({
      name: "MCP",
      version: "0.1.0",
    });
    client.setRequestHandler(z.object({ method: z.literal("whoami") }), () => {
      return tracer.startActiveSpan("whoami", (span) => {
        try {
          return {
            name: "OpenInference",
          };
        } finally {
          span.end();
        }
      });
    });
    await client.connect(transport);
    try {
      await tracer.startActiveSpan("root", async (span) => {
        try {
          const { tools } = await client.listTools();
          expect(tools).toHaveLength(1);
          expect(tools[0].name).toBe("hello");

          const res = (await client.callTool({ name: "hello" })) as {
            content: Array<{ type: string; text: string }>;
          };
          expect(res.content).toHaveLength(1);
          expect(res.content[0].text).toBe("Hello OpenInference!");
        } finally {
          span.end();
        }
      });
    } finally {
      await client.close();
    }
    await tracerProvider.forceFlush();
    // There does not seem to be a reliable way to wait for the child to exit completely with StdioClientTransport
    // so poll for the expected telemetry length instead.
    for (let i = 0; i < 100; i++) {
      if (telemetry.resourceSpans.length >= 2) {
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    expect(telemetry.resourceSpans).toHaveLength(2);
    let rootSpan: ExportedSpan | undefined;
    let serverSpan: ExportedSpan | undefined;
    let whoamiSpan: ExportedSpan | undefined;
    for (const resourceSpan of telemetry.resourceSpans) {
      expect(resourceSpan.scopeSpans).toHaveLength(1);
      for (const scopeSpan of resourceSpan.scopeSpans) {
        if (scopeSpan.scope.name === "mcp-test-client") {
          for (const span of scopeSpan.spans) {
            if (span.name === "root") {
              rootSpan = span;
            } else if (span.name === "whoami") {
              whoamiSpan = span;
            }
          }
        } else if (scopeSpan.scope.name === "mcp-test-server") {
          serverSpan = scopeSpan.spans[0];
        }
      }
    }
    expect(rootSpan).toBeDefined();
    expect(rootSpan!.name).toBe("root");
    expect(serverSpan).toBeDefined();
    expect(serverSpan!.name).toBe("hello");
    expect(whoamiSpan).toBeDefined();
    expect(whoamiSpan!.name).toBe("whoami");
    expect(serverSpan!.traceId).toBe(rootSpan!.traceId);
    expect(serverSpan!.parentSpanId).toBe(rootSpan!.spanId);
    expect(whoamiSpan!.traceId).toBe(rootSpan!.traceId);
    expect(whoamiSpan!.parentSpanId).toBe(serverSpan!.spanId);
  });
  it("propagates context - sse", async () => {
    const child = spawn("tsx", [path.join(__dirname, "mcpserver.ts")], {
      env: {
        ...process.env,
        MCP_TRANSPORT: "sse",
        OTEL_EXPORTER_OTLP_ENDPOINT: otlpEndpoint,
      },
      stdio: ["ignore", "pipe", "inherit"],
    });
    const childClosedPromise = new Promise<void>((resolve) => {
      child.on("close", () => {
        resolve();
      });
    });
    const sseUrl = await new Promise<string>((resolve) => {
      child.stdout.on("data", (data: Buffer) => {
        const line = data.toString().trim();
        if (line.startsWith("Server running on ")) {
          const endpoint = line.substring("Server running on ".length);
          resolve(endpoint);
        }
      });
    });
    try {
      const transport = new SSEClientTransport(new URL(sseUrl));
      const client = new MCPClientModule.Client({
        name: "MCP",
        version: "0.1.0",
      });
      client.setRequestHandler(
        z.object({ method: z.literal("whoami") }),
        () => {
          return tracer.startActiveSpan("whoami", (span) => {
            try {
              return {
                name: "OpenInference",
              };
            } finally {
              span.end();
            }
          });
        },
      );
      await client.connect(transport);
      try {
        await tracer.startActiveSpan("root", async (span) => {
          try {
            const { tools } = await client.listTools();
            expect(tools).toHaveLength(1);
            expect(tools[0].name).toBe("hello");

            const res = (await client.callTool({ name: "hello" })) as {
              content: Array<{ type: string; text: string }>;
            };
            expect(res.content).toHaveLength(1);
            expect(res.content[0].text).toBe("Hello OpenInference!");
          } finally {
            span.end();
          }
        });
      } finally {
        await client.close();
      }
    } finally {
      child.kill();
    }
    await tracerProvider.forceFlush();
    await childClosedPromise;
    expect(telemetry.resourceSpans).toHaveLength(2);
    let rootSpan: ExportedSpan | undefined;
    let serverSpan: ExportedSpan | undefined;
    let whoamiSpan: ExportedSpan | undefined;
    for (const resourceSpan of telemetry.resourceSpans) {
      expect(resourceSpan.scopeSpans).toHaveLength(1);
      for (const scopeSpan of resourceSpan.scopeSpans) {
        if (scopeSpan.scope.name === "mcp-test-client") {
          for (const span of scopeSpan.spans) {
            if (span.name === "root") {
              rootSpan = span;
            } else if (span.name === "whoami") {
              whoamiSpan = span;
            }
          }
        } else if (scopeSpan.scope.name === "mcp-test-server") {
          serverSpan = scopeSpan.spans[0];
        }
      }
    }
    expect(rootSpan).toBeDefined();
    expect(rootSpan!.name).toBe("root");
    expect(serverSpan).toBeDefined();
    expect(serverSpan!.name).toBe("hello");
    expect(whoamiSpan).toBeDefined();
    expect(whoamiSpan!.name).toBe("whoami");
    expect(serverSpan!.traceId).toBe(rootSpan!.traceId);
    expect(serverSpan!.parentSpanId).toBe(rootSpan!.spanId);
    expect(whoamiSpan!.traceId).toBe(rootSpan!.traceId);
    expect(whoamiSpan!.parentSpanId).toBe(serverSpan!.spanId);
  });
  it("propagates context - streamable http", async () => {
    const child = spawn("tsx", [path.join(__dirname, "mcpserver.ts")], {
      env: {
        ...process.env,
        MCP_TRANSPORT: "streamableHttp",
        OTEL_EXPORTER_OTLP_ENDPOINT: otlpEndpoint,
      },
      stdio: ["ignore", "pipe", "inherit"],
    });
    const childClosedPromise = new Promise<void>((resolve) => {
      child.on("close", () => {
        resolve();
      });
    });
    const serverUrl = await new Promise<string>((resolve) => {
      child.stdout.on("data", (data: Buffer) => {
        const line = data.toString().trim();
        if (line.startsWith("Server running on ")) {
          const endpoint = line.substring("Server running on ".length);
          resolve(endpoint);
        }
      });
    });
    try {
      const transport = new StreamableHTTPClientTransport(new URL(serverUrl));
      const client = new MCPClientModule.Client({
        name: "MCP",
        version: "0.1.0",
      });
      client.setRequestHandler(
        z.object({ method: z.literal("whoami") }),
        () => {
          return tracer.startActiveSpan("whoami", (span) => {
            try {
              return {
                name: "OpenInference",
              };
            } finally {
              span.end();
            }
          });
        },
      );
      await client.connect(transport);
      try {
        await tracer.startActiveSpan("root", async (span) => {
          try {
            const { tools } = await client.listTools();
            expect(tools).toHaveLength(1);
            expect(tools[0].name).toBe("hello");

            const res = (await client.callTool({ name: "hello" })) as {
              content: Array<{ type: string; text: string }>;
            };
            expect(res.content).toHaveLength(1);
            expect(res.content[0].text).toBe("Hello OpenInference!");
          } finally {
            span.end();
          }
        });
      } finally {
        await client.close();
      }
    } finally {
      child.kill();
    }
    await tracerProvider.forceFlush();
    await childClosedPromise;
    expect(telemetry.resourceSpans).toHaveLength(2);
    let rootSpan: ExportedSpan | undefined;
    let serverSpan: ExportedSpan | undefined;
    let whoamiSpan: ExportedSpan | undefined;
    for (const resourceSpan of telemetry.resourceSpans) {
      expect(resourceSpan.scopeSpans).toHaveLength(1);
      for (const scopeSpan of resourceSpan.scopeSpans) {
        if (scopeSpan.scope.name === "mcp-test-client") {
          for (const span of scopeSpan.spans) {
            if (span.name === "root") {
              rootSpan = span;
            } else if (span.name === "whoami") {
              whoamiSpan = span;
            }
          }
        } else if (scopeSpan.scope.name === "mcp-test-server") {
          serverSpan = scopeSpan.spans[0];
        }
      }
    }
    expect(rootSpan).toBeDefined();
    expect(rootSpan!.name).toBe("root");
    expect(serverSpan).toBeDefined();
    expect(serverSpan!.name).toBe("hello");
    expect(whoamiSpan).toBeDefined();
    expect(whoamiSpan!.name).toBe("whoami");
    expect(serverSpan!.traceId).toBe(rootSpan!.traceId);
    expect(serverSpan!.parentSpanId).toBe(rootSpan!.spanId);
    expect(whoamiSpan!.traceId).toBe(rootSpan!.traceId);
    expect(whoamiSpan!.parentSpanId).toBe(serverSpan!.spanId);
  });
});
