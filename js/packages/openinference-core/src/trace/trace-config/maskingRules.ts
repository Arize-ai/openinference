import { AttributeValue } from "@opentelemetry/api";
import { REDACTED_VALUE } from "./constants";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";
import { MaskingRule, MaskingRuleArgs } from "./types";

/**
 * Masks (redacts) input text in LLM input messages.
 * Will mask information stored under the key `llm.input_messages.[i].message.content`.
 * @example
 * ```typescript
 *  maskInputTextRule.condition({
 *      config: {hideInputText: true},
 *      key: "llm.input_messages.[i].message.content"
 *  }) // returns true so the rule applies and the value will be redacted
 */
const maskInputTextRule: MaskingRule = {
  condition: ({ config, key }) =>
    config.hideInputText &&
    key.includes(SemanticConventions.LLM_INPUT_MESSAGES) &&
    key.includes(SemanticConventions.MESSAGE_CONTENT) &&
    !key.includes(SemanticConventions.MESSAGE_CONTENTS),
  action: () => REDACTED_VALUE,
};

/**
 * Masks (redacts) output text in LLM output messages.
 * Will mask information stored under the key `llm.output_messages.[i].message.content`.
 * @example
 * ```typescript
 *  maskOutputTextRule.condition({
 *      config: {hideOutputText: true},
 *      key: "llm.output_messages.[i].message.content"
 *  }) // returns true so the rule applies and the value will be redacted
 * ```
 */
const maskOutputTextRule: MaskingRule = {
  condition: ({ config, key }) =>
    config.hideOutputText &&
    key.includes(SemanticConventions.LLM_OUTPUT_MESSAGES) &&
    key.includes(SemanticConventions.MESSAGE_CONTENT) &&
    !key.includes(SemanticConventions.MESSAGE_CONTENTS),
  action: () => REDACTED_VALUE,
};

/**
 * Masks (redacts) input text content in LLM input messages.
 * Will mask information stored under the key `llm.input_messages.[i].message.contents.[j].message_content.text`.
 * @example
 * ```typescript
 *  maskOutputTextRule.condition({
 *      config: {hideInputText: true},
 *      key: "llm.input_messages.[i].message.contents.[j].message_content.text"
 *  }) // returns true so the rule applies and the value will be redacted
 */
const maskInputTextContentRule: MaskingRule = {
  condition: ({ config, key }) =>
    config.hideInputText &&
    key.includes(SemanticConventions.LLM_INPUT_MESSAGES) &&
    key.includes(SemanticConventions.MESSAGE_CONTENT_TEXT),
  action: () => REDACTED_VALUE,
};

/**
 * Masks (redacts) output text content in LLM output messages.
 * @example
 * ```typescript
 *  maskOutputTextRule.condition({
 *      config: {hideOutputText: true},
 *      key: "llm.output_messages.[i].message.contents.[j].message_content.text"
 *  }) // returns true so the rule applies and the value will be redacted
 */
const maskOutputTextContentRule: MaskingRule = {
  condition: ({ config, key }) =>
    config.hideOutputText &&
    key.includes(SemanticConventions.LLM_OUTPUT_MESSAGES) &&
    key.includes(SemanticConventions.MESSAGE_CONTENT_TEXT),
  action: () => REDACTED_VALUE,
};

/**
 * Masks (removes) input images in LLM input messages.
 * @example
 * ```typescript
 *  maskOutputTextRule.condition({
 *      config: {hideInputImages: true},
 *      key: "llm.input_messages.[i].message.contents.[j].message_content.image"
 *  }) // returns true so the rule applies and the value will be removed
 */
const maskInputImagesRule: MaskingRule = {
  condition: ({ config, key }) =>
    config.hideInputImages &&
    key.includes(SemanticConventions.LLM_INPUT_MESSAGES) &&
    key.includes(SemanticConventions.MESSAGE_CONTENT_IMAGE),
  action: () => undefined,
};

function isBase64Url(url?: AttributeValue): boolean {
  return (
    typeof url === "string" &&
    url.startsWith("data:image/") &&
    url.includes("base64")
  );
}

/**
 * Masks (redacts) base64 images that are too long.
 *  * @example
 * ```typescript
 *  maskOutputTextRule.condition({
 *      config: {base64ImageMaxLength: 10},
 *      key: "llm.input_messages.[i].message.contents.[j].message_content.image.url",
 *      value: "data:image/base64,verylongbase64string"
 *  }) // returns true so the rule applies and the value will be redacted
 */
const maskLongBase64ImageRule: MaskingRule = {
  condition: ({ config, key, value }) =>
    typeof value === "string" &&
    isBase64Url(value) &&
    value.length > config.base64ImageMaxLength &&
    key.includes(SemanticConventions.LLM_INPUT_MESSAGES) &&
    key.includes(SemanticConventions.MESSAGE_CONTENT_IMAGE) &&
    key.endsWith(SemanticConventions.IMAGE_URL),
  action: () => REDACTED_VALUE,
};

/**
 * Masks (removes) embedding vectors.
 *  * @example
 * ```typescript
 *  maskOutputTextRule.condition({
 *      config: {hideEmbeddingVectors: true},
 *      key: "embedding.embeddings.[i].embedding.vector"
 *  }) // returns true so the rule applies and the value will be redacted
 */
const maskEmbeddingVectorsRule: MaskingRule = {
  condition: ({ config, key }) =>
    config.hideEmbeddingVectors &&
    key.includes(SemanticConventions.EMBEDDING_EMBEDDINGS) &&
    key.includes(SemanticConventions.EMBEDDING_VECTOR),
  action: () => undefined,
};

/**
 * A list of {@link MaskingRule}s that are applied to span attributes to either redact or remove sensitive information.
 * The order of these rules is important as it can ensure appropriate masking of information
 * Rules should go from more specific to more general so that things like `llm.input_messages.[i].message.content` are masked with {@link REDACTED_VALUE} before the more generic masking of `llm.input_messages` might happen with `undefined` might happen.
 */
const maskingRules: MaskingRule[] = [
  {
    condition: ({ config, key }) =>
      config.hideInputs && key === SemanticConventions.INPUT_VALUE,
    action: () => REDACTED_VALUE,
  },
  {
    condition: ({ config, key }) =>
      config.hideInputs && key === SemanticConventions.INPUT_MIME_TYPE,
    action: () => undefined,
  },
  {
    condition: ({ config, key }) =>
      config.hideOutputs && key === SemanticConventions.OUTPUT_VALUE,
    action: () => REDACTED_VALUE,
  },
  {
    condition: ({ config, key }) =>
      config.hideOutputs && key === SemanticConventions.OUTPUT_MIME_TYPE,
    action: () => undefined,
  },
  {
    condition: ({ config, key }) =>
      (config.hideInputs || config.hideInputMessages) &&
      key.includes(SemanticConventions.LLM_INPUT_MESSAGES),
    action: () => undefined,
  },
  {
    condition: ({ config, key }) =>
      (config.hideOutputs || config.hideOutputMessages) &&
      key.includes(SemanticConventions.LLM_OUTPUT_MESSAGES),
    action: () => undefined,
  },
  maskInputTextRule,
  maskOutputTextRule,
  maskInputTextContentRule,
  maskOutputTextContentRule,
  maskInputImagesRule,
  maskLongBase64ImageRule,
  maskEmbeddingVectorsRule,
];

/**
 * A function that masks (redacts or removes) sensitive information from span attributes based on the trace config.
 * @param config The {@link TraceConfig} to use to determine if the value should be masked
 * @param key The key of the attribute to mask
 * @param value The value of the attribute to mask
 * @returns The redacted value or undefined if the value should be masked, otherwise the original value
 */
export function mask({
  config,
  key,
  value,
}: MaskingRuleArgs): AttributeValue | undefined {
  for (const rule of maskingRules) {
    if (rule.condition({ config, key, value })) {
      return rule.action();
    }
  }
  return value;
}
