import { Mastra } from "@mastra/core/mastra";
import { phoenixAgent } from "./agents";
import {
  isOpenInferenceSpan,
  OpenInferenceOTLPTraceExporter,
} from "@arizeai/openinference-mastra";
import { env } from "./env";
import { storage } from "./storage";

export const mastra = new Mastra({
  agents: {
    phoenixAgent: await phoenixAgent({
      apiKey: env.PHOENIX_API_KEY,
      apiUrl: env.PHOENIX_API_URL,
    }),
  },
  storage,
  telemetry: {
    enabled: true,
    serviceName: "phoenix-agent",
    export: {
      type: "custom",
      exporter: new OpenInferenceOTLPTraceExporter({
        headers: {
          Authorization: `Bearer ${env.PHOENIX_API_KEY}`,
        },
        url: `${env.PHOENIX_API_URL}/v1/traces`,
        spanFilter: isOpenInferenceSpan,
      }),
    },
  },
});
