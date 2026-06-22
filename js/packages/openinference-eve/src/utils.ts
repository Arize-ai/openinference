import type { Attributes, AttributeValue } from "@opentelemetry/api";
import { diag } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";

import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { EVE_ATTRIBUTE_KEYS, EVE_ATTRIBUTE_PREFIX, EveFunctionNameToSpanKindMap } from "./constants";

const getEveTurnSpanKind = (span: ReadableSpan): string | undefined => {
  const operationName = span.attributes["operation.name"];
  const name = typeof operationName === "string" ? operationName : span.name;
  // operation.name may include a user-provided functionId suffix after a space
  const functionName = name.split(" ")[0];
  return EveFunctionNameToSpanKindMap.get(functionName);
};

const getEveConvertedAttributes = (attributes: Attributes): Attributes => {
  const result: Attributes = {};

  const sessionId = attributes[EVE_ATTRIBUTE_KEYS.SESSION_ID];
  if (typeof sessionId === "string") {
    result[SemanticConventions.SESSION_ID] = sessionId;
  }

  // Map remaining eve.* attributes to metadata.*
  for (const [key, value] of Object.entries(attributes)) {
    if (key.startsWith(EVE_ATTRIBUTE_PREFIX) && key !== EVE_ATTRIBUTE_KEYS.SESSION_ID) {
      result[`${SemanticConventions.METADATA}.${key}`] = value;
    }
  }

  return result;
};

/**
 * Adds OpenInference attributes derived from Eve AI framework span attributes.
 *
 * For spans carrying `eve.*` attributes this function:
 * - Maps `eve.session.id` → `session.id`
 * - Maps all other `eve.*` attributes → `metadata.eve.*`
 * - Sets `openinference.span.kind` = AGENT on `ai.eve.turn` root spans
 *
 * This runs before the Vercel/GenAI processing so the span kind is respected
 * by the existing `getOISpanKindFromAttributes` guard ("if already set, use it").
 */
export const addEveAttributesToSpan = (span: ReadableSpan): void => {
  const hasEveAttributes = Object.keys(span.attributes).some((key) =>
    key.startsWith(EVE_ATTRIBUTE_PREFIX),
  );

  if (!hasEveAttributes) {
    return;
  }

  try {
    const attrs = span.attributes as Record<string, AttributeValue>;
    const newAttrs = getEveConvertedAttributes(span.attributes);

    const spanKind = getEveTurnSpanKind(span);
    if (spanKind != null && span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] == null) {
      newAttrs[SemanticConventions.OPENINFERENCE_SPAN_KIND] = spanKind;
    }

    Object.entries(newAttrs).forEach(([key, value]) => {
      attrs[key] = value as AttributeValue;
    });
  } catch (error) {
    diag.warn(`Unable to add OpenInference Eve attributes to span: ${error}`);
  }
};
