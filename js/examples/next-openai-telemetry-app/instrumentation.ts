import { registerOTel } from "@vercel/otel";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { OpenInferenceSpanProcessor } from "@arizeai/openinference-vercel";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import assert from "assert";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

assert(process.env["PHOENIX_API_KEY"], "PHOENIX_API_KEY is required");

export function register() {
  registerOTel({
    serviceName: "next-app",
    spanProcessors: [
      new OpenInferenceSpanProcessor(),
      new SimpleSpanProcessor(
        new OTLPTraceExporter({
          headers: {
            api_key: process.env["PHOENIX_API_KEY"],
          },
          url: "https://app.phoenix.arize.com/v1/traces",
        }),
      ),
    ],
  });
}
