import otelGenaiClientSpanToolCalls from "./__fixtures__/otel_genai_client_span_tool_calls.json";
import openinferenceClientSpanToolCalls from "./__fixtures__/openinference_client_span_tool_calls.json";
import { convertGenAISpanAttributesToOpenInferenceSpanAttributes } from "../src/attributes.js";

describe("convertGenAISpanAttributesToOpenInferenceSpanAttributes", () => {
  it("should convert OpenAI client span attributes to OpenInference span attributes", () => {
    const convertedAttributes =
      convertGenAISpanAttributesToOpenInferenceSpanAttributes(
        otelGenaiClientSpanToolCalls,
      );
    expect(convertedAttributes).toEqual(openinferenceClientSpanToolCalls);
  });
});
