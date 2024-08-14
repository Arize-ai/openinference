import { registerOTel } from "@vercel/otel";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { OpenInferenceSpanProcessor } from "../../packages/openinference-vercel-ai-sdk-span-processor/src/OpenInferenceSpanProcessor";
// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

export function register() {
  registerOTel({
    serviceName: "next-app",
    spanProcessors: [
      new OpenInferenceSpanProcessor(),
      new SimpleSpanProcessor(
        new OTLPTraceExporter({ url: "http://localhost:6006/v1/traces" }),
      ),
    ],
  });
}
