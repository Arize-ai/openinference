import * as otelSemanticConventions from "@opentelemetry/semantic-conventions/incubating";
import { describe, expect, it } from "vitest";

import {
  GenAIAttributes,
  GenAIOperationName,
  GenAIOutputType,
  GenAIProviderName,
} from "../../../src/trace/genai/genAIAttributes";

const otelExports: Record<string, unknown> = otelSemanticConventions;

/**
 * The GenAI constants are copied from the OpenTelemetry semantic conventions
 * instead of imported at runtime. These tests fail when a copy drifts from the
 * installed `@opentelemetry/semantic-conventions` package.
 */
describe("GenAI semantic convention constants", () => {
  const registryBackedConstants = [
    { name: "GenAIAttributes", constants: GenAIAttributes, otelPrefix: "ATTR_" },
    {
      name: "GenAIOperationName",
      constants: GenAIOperationName,
      otelPrefix: "GEN_AI_OPERATION_NAME_VALUE_",
    },
    {
      name: "GenAIProviderName",
      constants: GenAIProviderName,
      otelPrefix: "GEN_AI_PROVIDER_NAME_VALUE_",
    },
    {
      name: "GenAIOutputType",
      constants: GenAIOutputType,
      otelPrefix: "GEN_AI_OUTPUT_TYPE_VALUE_",
    },
  ];

  describe.each(registryBackedConstants)("$name", ({ constants, otelPrefix }) => {
    it.each(Object.entries(constants))(
      `should match the OpenTelemetry export for %s`,
      (key, value) => {
        expect(otelExports[`${otelPrefix}${key}`]).toBe(value);
      },
    );
  });
});
