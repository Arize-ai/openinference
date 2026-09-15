// Allow running examples from inside a Claude Code session.
// The SDK refuses to spawn a nested `claude` process when this is set.
delete process.env.CLAUDECODE;

import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { Resource } from "@opentelemetry/resources";
import { ConsoleSpanExporter } from "@opentelemetry/sdk-trace-base";
import {
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";

import { ClaudeAgentSDKInstrumentation } from "../src/index";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

const provider = new NodeTracerProvider({
  resource: new Resource({
    [ATTR_SERVICE_NAME]: "claude-agent-sdk-service",
    [SEMRESATTRS_PROJECT_NAME]: "claude-agent-sdk-service",
  }),
  spanProcessors: [
    new SimpleSpanProcessor(new ConsoleSpanExporter()),
    new SimpleSpanProcessor(
      new OTLPTraceExporter({
        url: "http://localhost:6006/v1/traces",
      }),
    ),
  ],
});

const agentInstrumentation = new ClaudeAgentSDKInstrumentation();

registerInstrumentations({
  instrumentations: [agentInstrumentation],
});

provider.register();

// eslint-disable-next-line no-console
console.log("OpenInference initialized");
