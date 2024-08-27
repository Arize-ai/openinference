import { registerOTel } from "@vercel/otel";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { OpenInferenceSpanProcessor } from "@arizeai/openinference-vercel";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

export function register() {
  registerOTel({
    serviceName: "phoenix-next-app",
    attributes: {
      [SEMRESATTRS_PROJECT_NAME]: "phoenix-next-app",
    },
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
