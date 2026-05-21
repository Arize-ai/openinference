/**
 * Shared OpenTelemetry + OpenInference instrumentation setup for the
 * `@openai/agents` examples. Import this file via side-effect at the top of
 * any example (e.g. `import "./instrumentation";`) before importing
 * `@openai/agents`.
 *
 * Spans are printed to the terminal via ConsoleSpanExporter. To send them to
 * a collector instead (e.g. Arize Phoenix), swap in an OTLPTraceExporter:
 *
 * ```ts
 * import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
 * spanProcessors: [
 *   new SimpleSpanProcessor(new OTLPTraceExporter({
 *     url: "http://localhost:6006/v1/traces",
 *   })),
 * ],
 * ```
 */
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { ConsoleSpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

import { OpenAIAgentsInstrumentation } from "../src";

const provider = new NodeTracerProvider({
  spanProcessors: [new SimpleSpanProcessor(new ConsoleSpanExporter())],
});

registerInstrumentations({
  instrumentations: [new OpenAIAgentsInstrumentation()],
});

provider.register();

// eslint-disable-next-line no-console
console.log("OpenInference instrumentation for @openai/agents initialized");
