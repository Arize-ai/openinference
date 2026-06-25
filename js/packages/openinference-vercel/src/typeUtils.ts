import type { ReadableSpan, Span } from "@opentelemetry/sdk-trace-base";

export const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((v) => typeof v === "string");

/**
 * Heuristic for whether a span originated from the Vercel AI SDK (or carries OTel
 * GenAI conventions). Used to detect AI SDK traces and to decide which spans are
 * eligible to be re-rooted when their non-AI parent is filtered out.
 */
export const isLikelyAISDKSpan = (span: ReadableSpan | Span): boolean => {
  const attrs = span.attributes as Record<string, unknown> | undefined;
  const opName = attrs?.["operation.name"];
  const opId = attrs?.["ai.operationId"];

  if (typeof opName === "string" && opName.startsWith("ai.")) return true;
  if (typeof opId === "string" && opId.startsWith("ai.")) return true;

  // gen_ai.* indicates AI SDK v6+ GenAI spans. ai.* attribute keys (e.g.
  // `ai.telemetry.functionId`, `ai.settings.*`) mark framework wrapper spans that
  // carry AI SDK telemetry but whose `operation.name` is not `ai.*` — for example a
  // per-turn span like `ai.eve.turn` (operation.name `eve`). Recognizing these keeps
  // such a wrapper as the trace root instead of orphaning its AI children.
  return (
    attrs != null &&
    Object.keys(attrs).some((k) => k.startsWith("gen_ai.") || k.startsWith("ai."))
  );
};

const isObjectWithStringKeys = (value: unknown): value is Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  return Object.keys(value).every((key) => typeof key === "string");
};

export const isArrayOfObjects = (value: unknown): value is Record<string, unknown>[] =>
  Array.isArray(value) && value.every(isObjectWithStringKeys);

export const assertUnreachable = (x: never): never => {
  throw new Error("Unexpected value: " + x);
};

export type Mutable<T> = { -readonly [P in keyof T]: T[P] };

export type ValueOf<T> = T[keyof T];
