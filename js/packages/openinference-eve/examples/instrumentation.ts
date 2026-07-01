/**
 * instrumentation.ts — drop this file into the root of your Eve agent project.
 *
 * Sends OpenInference-formatted spans to a local Phoenix instance (or any
 * OTLP-compatible backend).
 *
 * Prereqs:
 *   pnpm add @arizeai/openinference-eve @opentelemetry/exporter-trace-otlp-proto
 *
 * Environment variables:
 *   PHOENIX_COLLECTOR_ENDPOINT  (default: http://localhost:6006/v1/traces)
 *   PHOENIX_PROJECT_NAME        (default: my-eve-agent)
 *   PHOENIX_API_KEY             (optional)
 */

import { defineInstrumentation } from "eve/instrumentation";
import { registerOTel } from "@vercel/otel";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

import { isOpenInferenceSpan, OpenInferenceSimpleSpanProcessor } from "@arizeai/openinference-eve";

const phoenixUrl =
  process.env["PHOENIX_COLLECTOR_ENDPOINT"] ?? "http://localhost:6006/v1/traces";

export default defineInstrumentation({
  setup: ({ agentName }) =>
    registerOTel({
      serviceName: agentName,
      resourceAttributes: {
        [SEMRESATTRS_PROJECT_NAME]:
          process.env["PHOENIX_PROJECT_NAME"] ?? agentName,
      },
      spanProcessors: [
        new OpenInferenceSimpleSpanProcessor({
          exporter: new OTLPTraceExporter({
            url: phoenixUrl,
            headers:
              process.env["PHOENIX_API_KEY"] != null
                ? { Authorization: `Bearer ${process.env["PHOENIX_API_KEY"]}` }
                : undefined,
          }),
          // Only export spans that carry OpenInference attributes
          // (ai.eve.turn, ai.streamText.doStream, ai.toolCall, etc.)
          spanFilter: isOpenInferenceSpan,
        }),
      ],
    }),
});
