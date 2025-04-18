import { spawn } from "child_process";
import http from "http";
import { AddressInfo } from "net";
import path from "path";

import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import * as MCPClientModule from "@modelcontextprotocol/sdk/client/index";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio";
import * as MCPServerModule from "@modelcontextprotocol/sdk/server/index";
import { Tracer } from "@opentelemetry/api";

import { ExportedSpan, startCollector, Telemetry } from "./collector";
import { isClientPatched, isServerPatched, MCPInstrumentation } from "../src";

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
      clientModule: MCPClientModule,
      serverModule: MCPServerModule,
    });
  });

  afterAll(async () => {
    instrumentation.disable();
    await tracerProvider.shutdown();

    await new Promise((reject, resolve) => {
      collector.close((err) => {
        if (err) {
          reject(err);
        } else {
          resolve(true);
        }
      });
    });
  });

  beforeEach(() => {
    telemetry.clear();
  });
  it("is patched", () => {
    expect(
      (MCPClientModule as { openInferencePatched?: boolean })
        .openInferencePatched,
    ).toBe(true);
    expect(isClientPatched()).toBe(true);
    expect(
      (MCPServerModule as { openInferencePatched?: boolean })
        .openInferencePatched,
    ).toBe(true);
    expect(isServerPatched()).toBe(true);
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
          expect(res.content[0].text).toBe("World!");
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
    let clientSpan: ExportedSpan | undefined;
    let serverSpan: ExportedSpan | undefined;
    for (const resourceSpan of telemetry.resourceSpans) {
      expect(resourceSpan.scopeSpans).toHaveLength(1);
      for (const scopeSpan of resourceSpan.scopeSpans) {
        expect(scopeSpan.spans).toHaveLength(1);
        if (scopeSpan.scope.name === "mcp-test-client") {
          clientSpan = scopeSpan.spans[0];
        } else if (scopeSpan.scope.name === "mcp-test-server") {
          serverSpan = scopeSpan.spans[0];
        }
      }
    }
    expect(clientSpan).toBeDefined();
    expect(clientSpan!.name).toBe("root");
    expect(serverSpan).toBeDefined();
    expect(serverSpan!.name).toBe("hello");
    expect(serverSpan!.traceId).toBe(clientSpan!.traceId);
    expect(serverSpan!.parentSpanId).toBe(clientSpan!.spanId);
  });
  it("propagates context - sse", async () => {
    const child = spawn("tsx", [path.join(__dirname, "mcpserver.ts")], {
      env: {
        ...process.env,
        MCP_TRANSPORT: "sse",
        OTEL_EXPORTER_OTLP_ENDPOINT: otlpEndpoint,
      },
      stdio: ["ignore", "pipe", "ignore"],
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
            expect(res.content[0].text).toBe("World!");
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
    let clientSpan: ExportedSpan | undefined;
    let serverSpan: ExportedSpan | undefined;
    for (const resourceSpan of telemetry.resourceSpans) {
      expect(resourceSpan.scopeSpans).toHaveLength(1);
      for (const scopeSpan of resourceSpan.scopeSpans) {
        expect(scopeSpan.spans).toHaveLength(1);
        if (scopeSpan.scope.name === "mcp-test-client") {
          clientSpan = scopeSpan.spans[0];
        } else if (scopeSpan.scope.name === "mcp-test-server") {
          serverSpan = scopeSpan.spans[0];
        }
      }
    }
    expect(clientSpan).toBeDefined();
    expect(clientSpan!.name).toBe("root");
    expect(serverSpan).toBeDefined();
    expect(serverSpan!.name).toBe("hello");
    expect(serverSpan!.traceId).toBe(clientSpan!.traceId);
    expect(serverSpan!.parentSpanId).toBe(clientSpan!.spanId);
  });
});
