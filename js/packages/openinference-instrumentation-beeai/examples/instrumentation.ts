import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { Resource } from "@opentelemetry/resources";
import {
  ConsoleSpanExporter,
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";

import { BeeAIInstrumentation } from "../src/index";

import * as beeaiFramework from "beeai-framework";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.INFO);

/////////////////// instrumentation ///////////////////
const provider = new NodeTracerProvider({
  resource: new Resource({
    [ATTR_SERVICE_NAME]: "beeai",
    [SEMRESATTRS_PROJECT_NAME]: "beeai-project",
  }),
  spanProcessors: [
    new SimpleSpanProcessor(new ConsoleSpanExporter()),
    new SimpleSpanProcessor(
      new OTLPTraceExporter({
        url: "http://localhost:6006/v1/traces",
      }),
    ),
  ],
});

const beeAIInstrumentation = new BeeAIInstrumentation();
beeAIInstrumentation.manuallyInstrument(beeaiFramework);

provider.register();

// eslint-disable-next-line no-console
console.log("👀 OpenInference initialized");
