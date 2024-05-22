import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { OpenAIInstrumentation } from "@arizeai/openinference-instrumentation-openai";
import * as OpenAI from "openai";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.INFO);

import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { Resource } from "@opentelemetry/resources";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

const provider = new NodeTracerProvider({
    resource: new Resource({
        ["openinference.project.name"]: "nextjs-chat",
    }),
});

provider.addSpanProcessor(
    new SimpleSpanProcessor(
        new OTLPTraceExporter({
            url:
                process.env.COLLECTOR_ENDPOINT ||
                "http://localhost:6006/v1/traces",
        })
    )
);

// OpenAI must be manually instrumented as it doesn't have a traditional module structure
const openAIInstrumentation = new OpenAIInstrumentation({});
openAIInstrumentation.manuallyInstrument(OpenAI);

registerInstrumentations({
    instrumentations: [openAIInstrumentation],
});

provider.register();

const ai = new OpenAI.OpenAI();

console.log("👀 OpenInference initialized");
