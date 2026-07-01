/**
 * instrumentation-with-context.ts — enrich spans with per-request context.
 *
 * The `step.started` event fires before each model call. The `runtimeContext`
 * you return is merged onto the model-call span and its children as OTel
 * attributes — useful for associating traces with users, tickets, or channels.
 *
 * The `isChannel` helper narrows the channel union to a specific type so
 * TypeScript knows which metadata fields are available.
 *
 * Prereqs:
 *   pnpm add @arizeai/openinference-eve @opentelemetry/exporter-trace-otlp-proto
 */

import { defineInstrumentation, isChannel } from "eve/instrumentation";
import { registerOTel } from "@vercel/otel";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";

import { isOpenInferenceSpan, OpenInferenceSimpleSpanProcessor } from "@arizeai/openinference-eve";

// Import your channel definitions (adjust path to match your project layout).
// import supportChannel from "./channels/support.js";

export default defineInstrumentation({
  setup: ({ agentName }) =>
    registerOTel({
      serviceName: agentName,
      spanProcessors: [
        new OpenInferenceSimpleSpanProcessor({
          exporter: new OTLPTraceExporter({
            url: process.env["PHOENIX_COLLECTOR_ENDPOINT"] ?? "http://localhost:6006/v1/traces",
          }),
          spanFilter: isOpenInferenceSpan,
        }),
      ],
    }),

  events: {
    "step.started"(input) {
      // Example: attach support-channel metadata when the channel matches.
      // if (!isChannel(input.channel, supportChannel)) {
      //   return undefined;
      // }
      // return {
      //   runtimeContext: {
      //     "support.channel_id": input.channel.metadata.channelId ?? "",
      //     "support.user_id": input.channel.metadata.triggeringUserId ?? "",
      //   },
      // };

      // Generic example: tag every span with the turn sequence number.
      return {
        runtimeContext: {
          "app.turn_sequence": input.turn.sequence,
          "app.step_index": input.step.index,
        },
      };
    },
  },
});
