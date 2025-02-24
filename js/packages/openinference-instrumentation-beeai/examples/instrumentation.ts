import {
  NodeTracerProvider,
  SimpleSpanProcessor,
  ConsoleSpanExporter,
} from "@opentelemetry/sdk-trace-node";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { Resource } from "@opentelemetry/resources";
import * as beeaiFramework from "beeai-framework";
import { BeeAIInstrumentation } from "../src/index.js";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

/////////////////// instrumentation part ///////////////////
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
