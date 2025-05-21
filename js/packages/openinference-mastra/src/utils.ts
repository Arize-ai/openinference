import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";
export { isOpenInferenceSpan } from "@arizeai/openinference-vercel/utils";

/**
 * Augments a span with OpenInference project resource attribute.
 *
 * This function will add additional attributes to the span, based on the span's resource attributes.
 *
 * @param span - The span to augment.
 */
export const addOpenInferenceProjectResourceAttributeSpan = (
  span: ReadableSpan,
) => {
  const attributes = span.resource.attributes;
  if (ATTR_SERVICE_NAME in attributes) {
    attributes[SEMRESATTRS_PROJECT_NAME] = attributes[ATTR_SERVICE_NAME];
  }
};
