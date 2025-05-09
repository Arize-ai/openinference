import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { isOpenInferenceSpan as isOpenInferenceSpanVercel } from "@arizeai/openinference-vercel/utils";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";
import { enrichBySpanKind } from "./attributes.js";

/**
 * Augments a Mastra span with OpenInference resource attributes.
 *
 * This function will add additional attributes to the span, based on the Mastra span's resource attributes.
 *
 * @param span - The Mastra span to augment.
 */
export const addOpenInferenceResourceAttributesToMastraSpan = (
  span: ReadableSpan,
) => {
  const attributes = span.resource.attributes;
  if (ATTR_SERVICE_NAME in attributes) {
    attributes[SEMRESATTRS_PROJECT_NAME] = attributes[ATTR_SERVICE_NAME];
  }
};

/**
 * Augments a Mastra span with OpenInference attributes.
 *
 * This function will add additional attributes to the span, based on the Mastra span's attributes.
 *
 * @param span - The Mastra span to augment.
 */
export const addOpenInferenceAttributesToMastraSpan = (
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  span: ReadableSpan,
) => {
  enrichBySpanKind(span);
};

/**
 * Checks if a span is an OpenInference span.
 *
 * This function will check if the span is an OpenInference annotated span.
 *
 * It can be used as a span filter for the OpenInferenceTraceExporter, to ensure that only OpenInference annotated spans are exported.
 *
 * @param span - The span to check.
 */
export const isOpenInferenceSpan = (span: ReadableSpan) => {
  // TODO: Implement Mastra span check in addition to Vercel span check
  return isOpenInferenceSpanVercel(span);
};
