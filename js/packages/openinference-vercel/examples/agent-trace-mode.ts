/**
 * agent-trace-mode.ts — instrumenting Vercel AI SDK agent frameworks.
 *
 * `agentTraceMode` produces a clean agent trace tree for the Vercel AI SDK and
 * any framework built on it (e.g. Eve, Mastra). For each trace it:
 *   - promotes the first `ai.*` span (including framework wrappers such as
 *     `ai.eve.turn`) to the trace root, clearing its non-AI parent (HTTP/fetch);
 *   - labels that root `AGENT` if it has no span kind, so wrapper spans survive
 *     the `isOpenInferenceSpan` filter;
 *   - propagates `session.id` from the active OpenInference context;
 *   - stamps the trace's earliest input / latest output onto the root when it
 *     has none of its own (e.g. a bare turn-wrapper span).
 *
 * A single `agentTraceMode: true` is the whole integration — no framework
 * package or hook. Frameworks should emit OpenInference-standard attributes
 * themselves: `session.id` via `setSession` on the context (propagated here),
 * and metadata via the AI SDK's `experimental_telemetry.metadata` (arrives as
 * `ai.telemetry.metadata.*`, converted to `metadata.*` automatically).
 */
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { Resource } from "@opentelemetry/resources";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

import { isOpenInferenceSpan, OpenInferenceSimpleSpanProcessor } from "../src";

const phoenixUrl =
  process.env["PHOENIX_COLLECTOR_ENDPOINT"] ?? "http://localhost:6006/v1/traces";

export const tracerProvider = new NodeTracerProvider({
  resource: new Resource({
    [SEMRESATTRS_PROJECT_NAME]:
      process.env["PHOENIX_PROJECT_NAME"] ?? "openinference-vercel-agent",
  }),
  spanProcessors: [
    new OpenInferenceSimpleSpanProcessor({
      exporter: new OTLPTraceExporter({ url: phoenixUrl }),
      spanFilter: isOpenInferenceSpan,
      agentTraceMode: true,
    }),
  ],
});

tracerProvider.register();
