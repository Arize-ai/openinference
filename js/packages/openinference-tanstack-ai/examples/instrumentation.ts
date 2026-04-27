import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { Resource } from "@opentelemetry/resources";
import { ConsoleSpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.INFO);

const phoenixUrl =
  process.env["PHOENIX_COLLECTOR_ENDPOINT"] ?? "http://localhost:6006/v1/traces";

export const tracerProvider = new NodeTracerProvider({
  resource: new Resource({
    [SEMRESATTRS_PROJECT_NAME]:
      process.env["PHOENIX_PROJECT_NAME"] ?? "openinference-tanstack-ai-examples",
  }),
  spanProcessors: [
    new SimpleSpanProcessor(new ConsoleSpanExporter()),
    new SimpleSpanProcessor(
      new OTLPTraceExporter({
        url: phoenixUrl,
        headers:
          process.env["PHOENIX_API_KEY"] == null
            ? undefined
            : {
                api_key: process.env["PHOENIX_API_KEY"],
                Authorization: `Bearer ${process.env["PHOENIX_API_KEY"]}`,
              },
      }),
    ),
  ],
});

tracerProvider.register();

// eslint-disable-next-line no-console
console.log("OpenInference TanStack AI example initialized");
// eslint-disable-next-line no-console
console.log(`Phoenix OTLP endpoint: ${phoenixUrl}`);
