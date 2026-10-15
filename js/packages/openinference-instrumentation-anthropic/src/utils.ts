import type { Attributes } from "@opentelemetry/api";
import { diag } from "@opentelemetry/api";

import type { SpanOutput } from "@arizeai/openinference-core";
import { safelyJSONStringify } from "@arizeai/openinference-core";
import { MimeType } from "@arizeai/openinference-semantic-conventions";

import { MODULE_NAME } from "./constants";

/**
 * Runs an attribute extractor so that a malformed request or response can never
 * make the instrumentation throw into the caller's code. A failure costs the
 * attributes it would have produced, and is reported through `diag`.
 */
export function safelyGetAttributes(getAttributes: () => Attributes): Attributes {
  try {
    return getAttributes();
  } catch (error) {
    diag.warn(`Failed to extract ${MODULE_NAME} span attributes`, error);
    return {};
  }
}

/**
 * Serializes a value as a JSON input or output, or nothing when it cannot be.
 */
export function toJSONOutput(value: unknown): SpanOutput | undefined {
  const json = safelyJSONStringify(value);
  return json == null ? undefined : { value: json, mimeType: MimeType.JSON };
}
