import { registerOTel } from "@vercel/otel";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { VercelSpanProcessor } from "../../packages/openinference-vercel/src";
import { OTLPHttpProtoTraceExporter } from "@vercel/otel";
import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

export function register() {
  registerOTel({
    serviceName: "next-app",
    spanProcessors: [
      new VercelSpanProcessor(),
      new SimpleSpanProcessor(
        new OTLPHttpProtoTraceExporter({
          url: "http://localhost:6006/v1/traces",
        }),
      ),
    ],
  });
}
