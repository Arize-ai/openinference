import {
  convertGenAISpanAttributesToOpenInferenceSpanAttributes as unsafeConvertGenAISpanAttributesToOpenInferenceSpanAttributes,
  convertGenAISpanToOpenInference as unsafeConvertGenAISpanToOpenInference,
  addOpenInferenceAttributesToSpan as unsafeAddOpenInferenceAttributesToSpan,
} from "./attributes.js";
import { withSafety } from "./utils.js";

export const convertGenAISpanAttributesToOpenInferenceSpanAttributes = withSafety({
  fn: unsafeConvertGenAISpanAttributesToOpenInferenceSpanAttributes,
  onError(error) {
    // eslint-disable-next-line no-console
    console.error(
      "Unable to convert GenAI span attributes to OpenInference span attributes",
      error,
    );
  },
});

export const convertGenAISpanToOpenInference = withSafety({
  fn: unsafeConvertGenAISpanToOpenInference,
  onError(error) {
    // eslint-disable-next-line no-console
    console.error("Unable to convert GenAI span to OpenInference span", error);
  },
});

export const addOpenInferenceAttributesToSpan = withSafety({
  fn: unsafeAddOpenInferenceAttributesToSpan,
  onError(error) {
    // eslint-disable-next-line no-console
    console.error("Unable to add OpenInference attributes to GenAI span", error);
  },
});

export {
  inferOpenInferenceSpanKindFromGenAI,
  mapFinishReason,
  mapGenAIMessageEvents,
} from "./attributes.js";
export type {
  ConvertGenAISpanOptions,
  FinishReasonStrategy,
  GenAISpanEvent,
  GenAISpanLike,
  MutableGenAISpanLike,
  ProviderMapping,
  SpanKindResolver,
} from "./attributes.js";
