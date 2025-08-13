import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";

const MASTRA_AGENT_SPAN_NAME_PREFIXES = [
  "agent",
  "mastra.getAgent",
  "post /api/agents",
];

/**
 * Add the OpenInference span kind to the given Mastra span.
 *
 * This function will add the OpenInference span kind to the given Mastra span.
 */
const addOpenInferenceSpanKind = (
  span: ReadableSpan,
  kind: OpenInferenceSpanKind,
) => {
  span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] = kind;
};

/**
 * Get the OpenInference span kind for the given Mastra span.
 *
 * This function will return the OpenInference span kind for the given Mastra span, if it has already been set.
 */
const getOpenInferenceSpanKind = (span: ReadableSpan) => {
  return span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] as
    | OpenInferenceSpanKind
    | undefined;
};

/**
 * Get the closest OpenInference span kind for the given Mastra span.
 *
 * This function will attempt to detect the closest OpenInference span kind for the given Mastra span,
 * based on the span's name and parent span ID.
 */
const getOpenInferenceSpanKindFromMastraSpan = (
  span: ReadableSpan,
): OpenInferenceSpanKind | null => {
  const oiKind = getOpenInferenceSpanKind(span);
  if (oiKind) {
    return oiKind;
  }
  const spanName = span.name.toLowerCase();
  if (
    MASTRA_AGENT_SPAN_NAME_PREFIXES.some((prefix) =>
      spanName.startsWith(prefix),
    )
  ) {
    return OpenInferenceSpanKind.AGENT;
  }
  return null;
};

/**
 * Enrich a Mastra span with OpenInference attributes.
 *
 * This function will add additional attributes to the span, based on the Mastra span's attributes.
 *
 * It will attempt to detect the closest OpenInference span kind for the given Mastra span, and then
 * enrich the span with the appropriate attributes based on the span kind and current attributes.
 *
 * @param span - The Mastra span to enrich.
 * @param shouldMarkAsAgent - Whether this span should be marked as an AGENT span
 */
export const addOpenInferenceAttributesToMastraSpan = (
  span: ReadableSpan,
  shouldMarkAsAgent: boolean = false,
) => {
  const kind = getOpenInferenceSpanKindFromMastraSpan(span);
  if (kind) {
    addOpenInferenceSpanKind(span, kind);
  }

  // Mark root spans as AGENT if requested and not already set
  if (
    span.parentSpanContext === undefined &&
    !getOpenInferenceSpanKind(span) &&
    shouldMarkAsAgent
  ) {
    addOpenInferenceSpanKind(span, OpenInferenceSpanKind.AGENT);
  }

  // Map Mastra threadId to OpenInference session ID
  // Only set SESSION_ID if it doesn't already exist to avoid overwriting existing values
  const threadId = span.attributes.threadId;
  if (
    threadId &&
    (typeof threadId === "string" || typeof threadId === "number") &&
    !span.attributes[SemanticConventions.SESSION_ID]
  ) {
    span.attributes[SemanticConventions.SESSION_ID] = threadId;
  }
};
