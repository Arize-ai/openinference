import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";

export const addOpenInferenceResourceAttributesToMastraSpan = (
  span: ReadableSpan,
) => {
  const attributes = span.resource.attributes;
  if (ATTR_SERVICE_NAME in attributes) {
    attributes[SEMRESATTRS_PROJECT_NAME] = attributes[ATTR_SERVICE_NAME];
  }
  // eslint-disable-next-line no-console
  console.log("attributes", attributes);
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
) => {};
