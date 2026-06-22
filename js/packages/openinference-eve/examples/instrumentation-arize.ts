/**
 * instrumentation-arize.ts — send Eve spans to Arize Cloud.
 *
 * Prereqs:
 *   pnpm add @arizeai/openinference-eve @opentelemetry/exporter-trace-otlp-proto
 *
 * Environment variables:
 *   ARIZE_SPACE_ID   your Arize space ID
 *   ARIZE_API_KEY    your Arize API key
 */

import { defineInstrumentation } from "eve/instrumentation";
import { registerOTel } from "@vercel/otel";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

import { isOpenInferenceSpan, OpenInferenceSimpleSpanProcessor } from "@arizeai/openinference-eve";

export default defineInstrumentation({
  setup: ({ agentName }) =>
    registerOTel({
      serviceName: agentName,
      resourceAttributes: {
        [SEMRESATTRS_PROJECT_NAME]: agentName,
        "model_id": agentName,
      },
      spanProcessors: [
        new OpenInferenceSimpleSpanProcessor({
          exporter: new OTLPTraceExporter({
            url: "https://otlp.arize.com/v1",
            headers: {
              "space_id": process.env["ARIZE_SPACE_ID"] ?? "",
              "api_key": process.env["ARIZE_API_KEY"] ?? "",
            },
          }),
          spanFilter: isOpenInferenceSpan,
        }),
      ],
    }),
});
