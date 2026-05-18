import type { Attributes } from "@opentelemetry/api";

import {
  convertGenAISpanToOpenInference,
  type ConvertGenAISpanOptions,
  type GenAISpanLike,
  type SpanKindResolver,
} from "@arizeai/openinference-genai";
import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

const TANSTACK_ITERATIONS_ATTRIBUTE = "tanstack.ai.iterations";

export type ConvertTanStackAISpanOptions = Omit<ConvertGenAISpanOptions, "spanKindResolver"> & {
  spanKindResolver?: SpanKindResolver;
};

export const tanStackAISpanKindResolver: SpanKindResolver = ({ attributes, defaultKind }) => {
  if (attributes[TANSTACK_ITERATIONS_ATTRIBUTE] != null) {
    return OpenInferenceSpanKind.AGENT;
  }
  return defaultKind;
};

export const convertTanStackAISpanToOpenInference = (
  span: GenAISpanLike,
  options: ConvertTanStackAISpanOptions = {},
): Attributes => {
  const convertedAttributes = convertGenAISpanToOpenInference(span, {
    ...options,
    spanKindResolver(input) {
      const tanStackDefaultKind = tanStackAISpanKindResolver(input);
      return options.spanKindResolver?.({ ...input, defaultKind: tanStackDefaultKind }) ?? tanStackDefaultKind;
    },
  });

  if (convertedAttributes == null) {
    return {};
  }

  if (
    convertedAttributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
      OpenInferenceSpanKind.AGENT &&
    typeof span.name === "string" &&
    convertedAttributes[SemanticConventions.AGENT_NAME] == null
  ) {
    convertedAttributes[SemanticConventions.AGENT_NAME] = span.name;
  }

  return convertedAttributes;
};
