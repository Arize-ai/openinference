/* eslint-disable no-console */
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { OpenAIInstrumentation } from "@arizeai/openinference-instrumentation-openai";
import {
  ConsoleSpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { Resource } from "@opentelemetry/resources";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { SemanticResourceAttributes } from "@opentelemetry/semantic-conventions";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

const provider = new NodeTracerProvider({
  resource: new Resource({
    [SemanticResourceAttributes.SERVICE_NAME]: "openai-service",
  }),
});

provider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
// provider.addSpanProcessor(
//   new SimpleSpanProcessor(
//     new OTLPTraceExporter({
//       url: "http://localhost:6006/v1/traces",
//     }),
//   ),
// );

debugger;

registerInstrumentations({
  instrumentations: [new OpenAIInstrumentation({})],
});

provider.register();

console.log("👀 OpenInference initialized");
