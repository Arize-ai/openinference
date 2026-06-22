import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";

/**
 * Maps Eve AI framework span names to OpenInference span kinds.
 * Eve creates an `ai.eve.turn` root span per conversational turn on top of
 * the standard Vercel AI SDK spans it uses internally.
 */
export const EveFunctionNameToSpanKindMap = new Map([
  ["ai.eve.turn", OpenInferenceSpanKind.AGENT],
]);

export const EVE_ATTRIBUTE_KEYS = {
  SESSION_ID: "eve.session.id",
  VERSION: "eve.version",
  ENVIRONMENT: "eve.environment",
  TURN_ID: "eve.turn.id",
  TURN_SEQUENCE: "eve.turn.sequence",
  STEP_INDEX: "eve.step.index",
  CHANNEL_KIND: "eve.channel.kind",
} as const;

export const EVE_ATTRIBUTE_PREFIX = "eve.";
