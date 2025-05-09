import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { ReadableSpan } from "@opentelemetry/sdk-trace-base";

const MASTRA_ROOT_SPAN_NAME_PREFIXES = ["post /api/agents"];

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
 * Detect the closest OpenInference span kind for the given Mastra span.
 *
 * This function will attempt to detect the closest OpenInference span kind for the given Mastra span,
 * based on the span's name and parent span ID.
 */
const detectOpenInferenceSpanKindFromMastraSpan = (
  span: ReadableSpan,
): OpenInferenceSpanKind | null => {
  const oiKind = getOpenInferenceSpanKind(span);
  if (oiKind) {
    return oiKind;
  }
  const spanName = span.name.toLowerCase();
  const hasParent = span.parentSpanId != null;
  if (
    // child spans with an agent prefix in their name are agent spans
    (hasParent &&
      MASTRA_AGENT_SPAN_NAME_PREFIXES.some((prefix) =>
        spanName.startsWith(prefix),
      )) ||
    // root spans with a root span prefix in their name are agent spans
    (!hasParent &&
      MASTRA_ROOT_SPAN_NAME_PREFIXES.some((prefix) =>
        spanName.startsWith(prefix),
      ))
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
 */
export const enrichBySpanKind = (span: ReadableSpan) => {
  if (getOpenInferenceSpanKind(span)) {
    // span has been processed already, skip
    return;
  }
  const kind = detectOpenInferenceSpanKindFromMastraSpan(span);
  if (kind) {
    addOpenInferenceSpanKind(span, kind);
  }
  switch (kind) {
    case OpenInferenceSpanKind.AGENT: {
      if (span.parentSpanId == null) {
        // add input and output attributes to the span
        // TODO: We need to collect the input and output from the children spans as we process them
        // and add them to the span attributes here
      }
      break;
    }
    default:
      break;
  }
};
