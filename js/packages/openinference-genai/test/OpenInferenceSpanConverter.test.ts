import otelGenaiClientSpanToolCalls from "./__fixtures__/otel_genai_client_span_tool_calls.json";
import openinferenceClientSpanToolCalls from "./__fixtures__/openinference_client_span_tool_calls.json";
import otelGenaiDeprecatedClientSpanToolCalls from "./__fixtures__/otel_genai_deprecated_client_span_tool_calls.json";
import { convertGenAISpanAttributesToOpenInferenceSpanAttributes } from "../src/attributes.js";

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
    expect(convertedAttributes).toEqual(openinferenceClientSpanToolCalls);
  });
});
