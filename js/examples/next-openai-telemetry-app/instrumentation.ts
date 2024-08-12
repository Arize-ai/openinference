import { registerOTel } from "@vercel/otel";
// import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-node";
// import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

export function register() {
  registerOTel({
    serviceName: "next-app",
    // spanProcessors: [
    //   new SimpleSpanProcessor(
    //     new OTLPTraceExporter({
    //       url:
    //         process.env.COLLECTOR_ENDPOINT || "http://localhost:6006/v1/traces",
    //     }),
    //   ),
    // ],
  });
}
