import { Attributes } from "@opentelemetry/api";

export const hasAIAttributes = (attributes: Attributes) => {
  return Object.keys(attributes).some((key) => key.startsWith("ai."));
};
