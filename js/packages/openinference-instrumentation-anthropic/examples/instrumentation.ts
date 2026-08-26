import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { Resource } from "@opentelemetry/resources";
import { ConsoleSpanExporter } from "@opentelemetry/sdk-trace-base";
import {
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";

import { AnthropicInstrumentation } from "../src/index";

const projectName = process.env.PHOENIX_PROJECT_NAME ?? "anthropic-service";
const collectorEndpoint =
  process.env.PHOENIX_COLLECTOR_ENDPOINT ?? "http://localhost:6006/v1/traces";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

export const provider = new NodeTracerProvider({
  resource: new Resource({
    [ATTR_SERVICE_NAME]: projectName,
    [SEMRESATTRS_PROJECT_NAME]: projectName,
  }),
  spanProcessors: [
    new SimpleSpanProcessor(new ConsoleSpanExporter()),
    new SimpleSpanProcessor(
      new OTLPTraceExporter({
        url: collectorEndpoint,
      }),
    ),
  ],
});

const instrumentation = new AnthropicInstrumentation();
registerInstrumentations({
  instrumentations: [instrumentation],
});

provider.register();

// eslint-disable-next-line no-console
console.log("👀 OpenInference initialized");
