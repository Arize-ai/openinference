import {
  ConsoleSpanExporter,
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { BedrockAgentInstrumentation } from "../src";
import { registerInstrumentations } from "@opentelemetry/instrumentation";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

/////////////////// instrumentation ///////////////////
const provider = new NodeTracerProvider({
  spanProcessors: [
    new SimpleSpanProcessor(new ConsoleSpanExporter()),
    new SimpleSpanProcessor(
      new OTLPTraceExporter({
        url: "http://localhost:6006/v1/traces",
      }),
    ),
  ],
});

const instrumentation = new BedrockAgentInstrumentation();
registerInstrumentations({
  instrumentations: [instrumentation],
});
provider.register();

// eslint-disable-next-line no-console
console.log("👀 OpenInference initialized");
