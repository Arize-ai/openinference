/* eslint-disable no-console */
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { ConsoleSpanExporter } from "@opentelemetry/sdk-trace-base";
import {
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";
import { Resource } from "@opentelemetry/resources";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { SemanticResourceAttributes } from "@opentelemetry/semantic-conventions";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { LangChainInstrumentation } from "@arizeai/openinference-instrumentation-langchain";
import * as lcCallbackManager from "@langchain/core/callbacks/manager";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

const provider = new NodeTracerProvider({
  resource: new Resource({
    [SemanticResourceAttributes.SERVICE_NAME]: "chat-service",
  }),
});

provider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
provider.addSpanProcessor(
  new SimpleSpanProcessor(
    new OTLPTraceExporter({
      url: process.env.COLLECTOR_ENDPOINT || "http://localhost:6006/v1/traces",
    }),
  ),
);

registerInstrumentations({
  instrumentations: [],
});

// LangChain must be manually instrumented as it doesn't have a traditional module structure
const lcInstrumentation = new LangChainInstrumentation();
lcInstrumentation.manuallyInstrument(lcCallbackManager);

provider.register();

console.log("👀 OpenInference initialized");
