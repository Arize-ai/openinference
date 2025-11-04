import { convertGenAISpanAttributesToOpenInferenceSpanAttributes } from "../src/attributes.js";

import openinferenceClientSpanInputOutputOnly from "./__fixtures__/openinference_client_span_input_output_only.json" with { type: "json" };
import openinferenceClientSpanToolCalls from "./__fixtures__/openinference_client_span_tool_calls.json" with { type: "json" };
import otelGenaiClientSpanToolCalls from "./__fixtures__/otel_genai_client_span_tool_calls.json" with { type: "json" };
import otelGenaiDeprecatedClientSpanToolCalls from "./__fixtures__/otel_genai_deprecated_client_span_tool_calls.json" with { type: "json" };

describe("convertGenAISpanAttributesToOpenInferenceSpanAttributes", () => {
  it("should convert GenAI OpenAI client span attributes to OpenInference span attributes", () => {
    const convertedAttributes =
      convertGenAISpanAttributesToOpenInferenceSpanAttributes(
        otelGenaiClientSpanToolCalls,
      );
    expect(convertedAttributes).toEqual(openinferenceClientSpanToolCalls);
  });
  it("should convert deprecated GenAI OpenAI client span attributes to OpenInference span attributes", () => {
    const convertedAttributes =
      convertGenAISpanAttributesToOpenInferenceSpanAttributes(
        otelGenaiDeprecatedClientSpanToolCalls,
      );
    expect(convertedAttributes).toEqual(openinferenceClientSpanInputOutputOnly);
  });
});
