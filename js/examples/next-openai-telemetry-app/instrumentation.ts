import { registerOTel } from "@vercel/otel";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

export function register() {
  registerOTel({
    serviceName: "next-app",
  });
}
