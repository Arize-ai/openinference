import { registerOTel } from "@vercel/otel";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OpenInferenceProtoTraceExporter } from "../../packages/openinference-vercel-ai-sdk-span-processor/src/OpenInferenceSpanExporter";
// For troubleshooting, set the log level to DiagLogLevel.DEBUG
// diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

export function register() {
  registerOTel({
    serviceName: "next-app",
    spanProcessors: [
      new SimpleSpanProcessor(
        new OpenInferenceProtoTraceExporter({
          url: "http://localhost:6006/v1/traces",
        }),
      ),
    ],
  });
}
