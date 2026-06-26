import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

import { OpenTelemetry } from "@ai-sdk/otel";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { Resource } from "@opentelemetry/resources";
import { ConsoleSpanExporter } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { registerTelemetry } from "ai";

import { isOpenInferenceSpan, OpenInferenceSimpleSpanProcessor } from "../src/index.js";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.INFO);

const phoenixUrl =
  process.env["PHOENIX_COLLECTOR_ENDPOINT"] ??
  "http://localhost:6006/v1/traces";

export const tracerProvider = new NodeTracerProvider({
  resource: new Resource({
    [SEMRESATTRS_PROJECT_NAME]:
      process.env["PHOENIX_PROJECT_NAME"] ?? "openinference-vercel-examples",
  }),
  spanProcessors: [
    // Local debugging
    new OpenInferenceSimpleSpanProcessor({
      exporter: new ConsoleSpanExporter(),
      spanFilter: isOpenInferenceSpan,
    }),
    // Export to local Phoenix
    new OpenInferenceSimpleSpanProcessor({
      exporter: new OTLPTraceExporter({
        url: phoenixUrl,
        headers:
          process.env["PHOENIX_API_KEY"] != null
            ? {
                api_key: process.env["PHOENIX_API_KEY"],
                Authorization: `Bearer ${process.env["PHOENIX_API_KEY"]}`,
              }
            : undefined,
      }),
      spanFilter: isOpenInferenceSpan,
    }),
  ],
});

tracerProvider.register();

registerTelemetry(
  new OpenTelemetry({
    usage: true,
    providerMetadata: true,
    embedding: true,
    reranking: true,
    runtimeContext: true,
    headers: true,
    toolChoice: true,
    schema: true,
  }),
);

// eslint-disable-next-line no-console
console.log(`\nOpenInference Vercel example initialized`);
// eslint-disable-next-line no-console
console.log(`Phoenix OTLP endpoint: ${phoenixUrl}`);
