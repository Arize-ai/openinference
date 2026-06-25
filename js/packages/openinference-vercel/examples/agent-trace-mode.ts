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
 * No framework-specific package is required. If a framework attaches custom
 * attributes you want mapped onto OpenInference conventions, use `preProcessSpan`
 * (see the Eve example below).
 */
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";

import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { Resource } from "@opentelemetry/resources";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

import { isOpenInferenceSpan, OpenInferenceSimpleSpanProcessor } from "../src";

const phoenixUrl =
  process.env["PHOENIX_COLLECTOR_ENDPOINT"] ?? "http://localhost:6006/v1/traces";

/**
 * Optional: map a framework's custom span attributes onto OpenInference
 * conventions. This example maps Eve's `eve.*` attributes — `eve.session.id`
 * becomes `session.id` and the rest become `metadata.eve.*`. Frameworks that
 * emit their session/metadata via the AI SDK's `ai.telemetry.metadata.*` need
 * no hook at all; it is already converted.
 */
const mapFrameworkAttributes = (span: ReadableSpan): void => {
  const attrs = span.attributes as Record<string, unknown>;
  for (const [key, value] of Object.entries(attrs)) {
    if (!key.startsWith("eve.")) continue;
    if (key === "eve.session.id") {
      attrs["session.id"] = value as never;
    } else {
      attrs[`metadata.${key}`] = value as never;
    }
  }
};

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
      // Drop `preProcessSpan` entirely if the framework emits no custom attributes.
      preProcessSpan: mapFrameworkAttributes,
    }),
  ],
});

tracerProvider.register();
