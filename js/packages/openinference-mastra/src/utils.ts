import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { ReadableSpan } from "@opentelemetry/sdk-trace-base";

/**
 * Augments a Mastra span with OpenInference attributes.
 *
 * This function will add additional attributes to the span, based on the Mastra span's attributes.
 *
 * @param span - The Mastra span to augment.
 */
export const addOpenInferenceAttributesToMastraSpan = (span: ReadableSpan) => {
  const attributes = span.attributes;
  if (attributes["componentName"]) {
    span.attributes[SEMRESATTRS_PROJECT_NAME] = attributes["componentName"];
  }
};
