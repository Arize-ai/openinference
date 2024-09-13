import { tracer, opentelemetry } from "dd-trace";
// import "dd-trace/init";
import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter as GrpcOTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-grpc";
import { OpenAIInstrumentation } from "@arizeai/openinference-instrumentation-openai";
// import * as OpenAI from "openai";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import {
  ConsoleSpanExporter,
  NodeTracerProvider,
} from "@opentelemetry/sdk-trace-node";
import { Resource } from "@opentelemetry/resources";
import { Metadata } from "@grpc/grpc-js";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

tracer.init();

// Arize specific - Create metadata and add your headers
const metadata = new Metadata();

// Your Arize Space and API Keys, which can be found in the UI
metadata.set("space_id", "");
metadata.set("api_key", "");

const provider = new NodeTracerProvider({
  resource: new Resource({
    // Arize specific - The name of a new or preexisting model you
    // want to export spans to
    model_id: "parker test",
    model_version: "1.0.0",
  }),
});

provider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
provider.addSpanProcessor(
  new SimpleSpanProcessor(
    new GrpcOTLPTraceExporter({
      url: "https://otlp.arize.com/v1",
      metadata,
    }),
  ),
);

registerInstrumentations({
  instrumentations: [new OpenAIInstrumentation({})],
});

provider.register();

diag.info("👀 OpenInference initialized");
