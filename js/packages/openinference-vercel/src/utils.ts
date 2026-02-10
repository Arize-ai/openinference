import {
  safelyJSONParse,
  safelyJSONStringify,
  withSafety,
} from "@arizeai/openinference-core";
import { convertGenAISpanAttributesToOpenInferenceSpanAttributes } from "@arizeai/openinference-genai";
import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { Attributes, AttributeValue, diag } from "@opentelemetry/api";
import { isAttributeValue } from "@opentelemetry/core";
import { ReadableSpan } from "@opentelemetry/sdk-trace-base";

import { VercelSDKFunctionNameToSpanKindMap } from "./constants";
import { OpenInferenceIOConventionKey, SpanFilter } from "./types";
import { isArrayOfObjects, isStringArray } from "./typeUtils";
import { VercelAISemanticConventions } from "./VercelAISemanticConventions";

const onErrorCallback = (attributeType: string) => (error: unknown) => {
  diag.warn(
    `Unable to get OpenInference ${attributeType} attributes from AI attributes falling back to null: ${error}`,
  );
};

/**
 * Gets the Vercel function name from the operation.name attribute
 * @param operationName - the operation name of the span
 * Operation names are set on Vercel spans as under the operation.name attribute with the
 * @example ai.generateText.doGenerate <functionId>
 * @returns the Vercel function name from the operation name or undefined if not found
 */
const getVercelFunctionNameFromOperationName = (
  operationName: string,
): string | undefined => {
  return operationName.split(" ")[0];
};

/**
 * Gets the OpenInference span kind that corresponds to the Vercel operation name.
 * This is more specific than the gen_ai.* operation detection.
 * @param attributes - The attributes of the span
 * @returns the OpenInference span kind associated with the attributes or undefined if not found
 */
const getOISpanKindFromAttributes = (
  attributes: Attributes,
): OpenInferenceSpanKind | string | undefined => {
  // If the span kind is already set, just use it
  const existingOISpanKind =
    attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND];
  if (existingOISpanKind != null && typeof existingOISpanKind === "string") {
    return existingOISpanKind;
  }
  const maybeOperationName = attributes["operation.name"];
  if (maybeOperationName == null || typeof maybeOperationName !== "string") {
    return;
  }
  const maybeFunctionName =
    getVercelFunctionNameFromOperationName(maybeOperationName);
  if (maybeFunctionName == null) {
    return;
  }
  return VercelSDKFunctionNameToSpanKindMap.get(maybeFunctionName);
};

/**
 * {@link getOISpanKindFromAttributes} wrapped in {@link withSafety} which will return null if any error is thrown
 */
const safelyGetOISpanKindFromAttributes = withSafety({
  fn: getOISpanKindFromAttributes,
  onError: onErrorCallback("span kind"),
});

/**
 * Takes the attributes from the span and accumulates the attributes that are prefixed with "ai.settings" to be used as the invocation parameters
 * This is a fallback if gen_ai.* invocation parameters are not present.
 * @param attributes the initial attributes of the span
 * @returns the OpenInference attributes associated with the invocation parameters
 */
const getInvocationParamAttributes = (attributes: Attributes) => {
  const settingAttributeKeys = Object.keys(attributes).filter((key) =>
    key.startsWith(VercelAISemanticConventions.SETTINGS),
  );
  if (settingAttributeKeys.length === 0) {
    return null;
  }
  const settingAttributes = settingAttributeKeys.reduce((acc, key) => {
    const keyParts = key.split(".");
    const paramKey = keyParts[keyParts.length - 1];
    acc[paramKey] = attributes[key];
    return acc;
  }, {} as Attributes);

  return {
    [SemanticConventions.LLM_INVOCATION_PARAMETERS]:
      safelyJSONStringify(settingAttributes) ?? undefined,
  };
};

/**
 * {@link getInvocationParamAttributes} wrapped in {@link withSafety} which will return null if any error is thrown
 */
const safelyGetInvocationParamAttributes = withSafety({
  fn: getInvocationParamAttributes,
  onError: onErrorCallback("invocation parameters"),
});

/**
 * Determines whether the value is a valid JSON string
 * @param value the value to check
 * @returns whether the value is a valid JSON string
 */
const isValidJsonString = (value?: AttributeValue) => {
  if (typeof value !== "string") {
    return false;
  }
  const parsed = safelyJSONParse(value);
  return typeof parsed === "object" && parsed !== null;
};

/**
 * Gets the mime type of the attribute value
 * @param value the attribute value to check
 * @returns the mime type of the value
 */
const getMimeTypeFromValue = (value?: AttributeValue) => {
  if (isValidJsonString(value)) {
    return MimeType.JSON;
  }
  return MimeType.TEXT;
};

/**
 * Gets OpenInference attributes associated with the IO
 * @param object.attributeValue the IO attribute value set by Vercel
 * @param object.OpenInferenceSemanticConventionKey the corresponding OpenInference semantic convention
 * @returns the OpenInference attributes associated with the IO value
 */
const getIOValueAttributes = ({
  attributeValue,
  OpenInferenceSemanticConventionKey,
}: {
  attributeValue?: AttributeValue;
  OpenInferenceSemanticConventionKey: OpenInferenceIOConventionKey;
}) => {
  const mimeTypeSemanticConvention =
    OpenInferenceSemanticConventionKey === SemanticConventions.INPUT_VALUE
      ? SemanticConventions.INPUT_MIME_TYPE
      : SemanticConventions.OUTPUT_MIME_TYPE;
  return {
    [OpenInferenceSemanticConventionKey]: attributeValue,
    [mimeTypeSemanticConvention]: getMimeTypeFromValue(attributeValue),
  };
};

/**
 * {@link getIOValueAttributes} wrapped in {@link withSafety} which will return null if any error is thrown
 */
const safelyGetIOValueAttributes = withSafety({
  fn: getIOValueAttributes,
  onError: onErrorCallback("input / output"),
});

/**
 * Formats an embedding attribute value (i.e., embedding text or vector) into the expected format
 * Vercel embedding vector attributes are stringified arrays, however, the OpenInference spec expects them to be un-stringified arrays
 * @param value the value to format (either embedding text or vector)
 * @returns the formatted value or the original value if it is not a string or cannot be parsed
 */
const formatEmbeddingValue = (value: AttributeValue) => {
  if (typeof value !== "string") {
    return value;
  }
  const parsedValue = safelyJSONParse(value);
  if (isAttributeValue(parsedValue) && parsedValue !== null) {
    return parsedValue;
  }
  return value;
};

/**
 * Takes the Vercel embedding attribute value and returns the OpenInference attributes
 * @param attributes the span attributes
 * @param spanKind the span kind
 * @returns the OpenInference attributes associated with the embedding
 */
const getEmbeddingAttributes = (
  attributes: Attributes,
  spanKind: OpenInferenceSpanKind | string | undefined,
): Attributes | null => {
  // Only process embeddings for EMBEDDING spans
  if (spanKind !== OpenInferenceSpanKind.EMBEDDING) {
    return null;
  }

  const EMBEDDING_PREFIX = SemanticConventions.EMBEDDING_EMBEDDINGS;
  const result: Attributes = {};

  // Handle single embedding text (ai.value)
  const embeddingText = attributes[VercelAISemanticConventions.EMBEDDING_TEXT];
  if (embeddingText != null) {
    if (typeof embeddingText === "string") {
      result[`${EMBEDDING_PREFIX}.0.${SemanticConventions.EMBEDDING_TEXT}`] =
        formatEmbeddingValue(embeddingText);
    }
  }

  // Handle multiple embedding texts (ai.values)
  const embeddingTexts =
    attributes[VercelAISemanticConventions.EMBEDDING_TEXTS];
  if (isStringArray(embeddingTexts)) {
    embeddingTexts.forEach((text, index) => {
      result[
        `${EMBEDDING_PREFIX}.${index}.${SemanticConventions.EMBEDDING_TEXT}`
      ] = formatEmbeddingValue(text);
    });
  }

  // Handle single embedding vector (ai.embedding)
  const embeddingVector =
    attributes[VercelAISemanticConventions.EMBEDDING_VECTOR];
  if (embeddingVector != null) {
    if (typeof embeddingVector === "string") {
      result[`${EMBEDDING_PREFIX}.0.${SemanticConventions.EMBEDDING_VECTOR}`] =
        formatEmbeddingValue(embeddingVector);
    }
  }

  // Handle multiple embedding vectors (ai.embeddings)
  const embeddingVectors =
    attributes[VercelAISemanticConventions.EMBEDDING_VECTORS];
  if (isStringArray(embeddingVectors)) {
    embeddingVectors.forEach((vector, index) => {
      result[
        `${EMBEDDING_PREFIX}.${index}.${SemanticConventions.EMBEDDING_VECTOR}`
      ] = formatEmbeddingValue(vector);
    });
  }

  return Object.keys(result).length > 0 ? result : null;
};

/**
 * {@link getEmbeddingAttributes} wrapped in {@link withSafety} which will return null if any error is thrown
 */
const safelyGetEmbeddingAttributes = withSafety({
  fn: getEmbeddingAttributes,
  onError: onErrorCallback("embedding"),
});

/**
 * Gets the input_messages OpenInference attributes from ai.prompt.messages
 * @param promptMessages the attribute value of the Vercel prompt messages
 * @returns input_messages attributes
 */
const getInputMessageAttributes = (promptMessages?: AttributeValue) => {
  if (typeof promptMessages !== "string") {
    return null;
  }

  const messages = safelyJSONParse(promptMessages);

  if (!isArrayOfObjects(messages)) {
    return null;
  }

  // Track the output message index separately since tool messages with multiple results
  // get expanded into multiple OpenInference messages
  let outputMessageIndex = 0;

  return messages.reduce((acc: Attributes, message) => {
    if (message.role === "tool") {
      const contentArray: unknown[] = Array.isArray(message.content)
        ? message.content
        : message.content
          ? [message.content]
          : [];

      // Per OpenInference spec, each tool result should be a separate message with:
      // - message.role: "tool"
      // - message.content: the result content
      // - message.tool_call_id: linking back to the original tool call
      // When Vercel sends multiple tool results in one message, we expand them.
      const toolResultAttributes = contentArray.reduce(
        (toolAcc: Attributes, content) => {
          if (typeof content !== "object" || content === null) {
            // bail out if the content is not an object
            return toolAcc;
          }
          if (!("output" in content) && !("result" in content)) {
            // bail out if the content does not have an output or result property
            return toolAcc;
          }
          const MESSAGE_PREFIX = `${SemanticConventions.LLM_INPUT_MESSAGES}.${outputMessageIndex}`;
          outputMessageIndex++;

          // Extract tool output from various possible formats:
          // 1. Newer AI SDK v6: content.output (the raw output value)
          // 2. Legacy format: content.result
          const TOOL_OUTPUT =
            "output" in content
              ? content.output
              : "result" in content
                ? content.result
                : undefined;
          const TOOL_OUTPUT_JSON =
            typeof TOOL_OUTPUT === "string"
              ? TOOL_OUTPUT
              : TOOL_OUTPUT != null
                ? (safelyJSONStringify(TOOL_OUTPUT) ?? undefined)
                : undefined;
          const TOOL_CALL_ID =
            "toolCallId" in content && typeof content.toolCallId === "string"
              ? content.toolCallId
              : undefined;
          const TOOL_NAME =
            "toolName" in content && typeof content.toolName === "string"
              ? content.toolName
              : undefined;
          return {
            ...toolAcc,
            [`${MESSAGE_PREFIX}.${SemanticConventions.MESSAGE_ROLE}`]: "tool",
            [`${MESSAGE_PREFIX}.${SemanticConventions.MESSAGE_CONTENT}`]:
              TOOL_OUTPUT_JSON,
            [`${MESSAGE_PREFIX}.${SemanticConventions.MESSAGE_TOOL_CALL_ID}`]:
              TOOL_CALL_ID,
            [`${MESSAGE_PREFIX}.${SemanticConventions.TOOL_NAME}`]: TOOL_NAME,
          };
        },
        {} as Attributes,
      );

      return {
        ...acc,
        ...toolResultAttributes,
      };
    }

    const MESSAGE_PREFIX = `${SemanticConventions.LLM_INPUT_MESSAGES}.${outputMessageIndex}`;
    outputMessageIndex++;

    if (isArrayOfObjects(message.content)) {
      const messageAttributes = message.content.reduce(
        (acc: Attributes, content, contentIndex) => {
          const CONTENTS_PREFIX = `${MESSAGE_PREFIX}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}`;
          const TOOL_CALL_PREFIX = `${MESSAGE_PREFIX}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${contentIndex}`;
          // prefer the input property over the args property
          // newer versions of ai-sdk use the input property instead of the args property
          const TOOL_CALL_ARGS =
            content.input != null
              ? content.input
              : content.args != null
                ? content.args
                : undefined;
          // Do not double-stringify the tool call arguments
          const TOOL_CALL_ARGS_JSON =
            typeof TOOL_CALL_ARGS === "string"
              ? TOOL_CALL_ARGS
              : TOOL_CALL_ARGS != null
                ? (safelyJSONStringify(TOOL_CALL_ARGS) ?? undefined)
                : undefined;
          return {
            ...acc,
            [`${CONTENTS_PREFIX}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`]:
              typeof content.type === "string" ? content.type : undefined,
            [`${CONTENTS_PREFIX}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`]:
              typeof content.text === "string" ? content.text : undefined,
            [`${CONTENTS_PREFIX}.${SemanticConventions.MESSAGE_CONTENT_IMAGE}`]:
              typeof content.image === "string" ? content.image : undefined,
            [`${TOOL_CALL_PREFIX}.${SemanticConventions.TOOL_CALL_ID}`]:
              typeof content.toolCallId === "string"
                ? content.toolCallId
                : undefined,
            [`${TOOL_CALL_PREFIX}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`]:
              typeof content.toolName === "string"
                ? content.toolName
                : undefined,
            [`${TOOL_CALL_PREFIX}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`]:
              TOOL_CALL_ARGS_JSON,
          };
        },
        {},
      );
      acc = {
        ...acc,
        ...messageAttributes,
      };
    } else if (typeof message.content === "string") {
      acc[`${MESSAGE_PREFIX}.${SemanticConventions.MESSAGE_CONTENT}`] =
        message.content;
    }
    acc[`${MESSAGE_PREFIX}.${SemanticConventions.MESSAGE_ROLE}`] =
      typeof message.role === "string" ? message.role : undefined;
    return acc;
  }, {});
};

/**
 * {@link getInputMessageAttributes} wrapped in {@link withSafety} which will return null if any error is thrown
 */
const safelyGetInputMessageAttributes = withSafety({
  fn: getInputMessageAttributes,
  onError: onErrorCallback("input message"),
});

/**
 * Gets the output_messages tool_call OpenInference attributes
 * @param toolCalls the attribute value of the Vercel ai.response.toolCalls
 * @returns output_messages tool_call attributes
 */
const getToolCallMessageAttributes = (toolCalls?: AttributeValue) => {
  if (typeof toolCalls !== "string") {
    return null;
  }

  const parsedToolCalls = safelyJSONParse(toolCalls);

  if (!isArrayOfObjects(parsedToolCalls)) {
    return null;
  }

  const OUTPUT_MESSAGE_PREFIX = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0`;
  return {
    [`${OUTPUT_MESSAGE_PREFIX}.${SemanticConventions.MESSAGE_ROLE}`]:
      "assistant",
    ...parsedToolCalls.reduce((acc: Attributes, toolCall, index) => {
      const TOOL_CALL_PREFIX = `${OUTPUT_MESSAGE_PREFIX}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${index}`;
      // newer versions of Vercel use the input property instead of the args property
      const toolCallArgsJSON =
        toolCall.args != null
          ? safelyJSONStringify(toolCall.args)
          : toolCall.input != null
            ? safelyJSONStringify(toolCall.input)
            : undefined;
      return {
        ...acc,
        [`${TOOL_CALL_PREFIX}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`]:
          isAttributeValue(toolCall.toolName) ? toolCall.toolName : undefined,
        [`${TOOL_CALL_PREFIX}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`]:
          toolCallArgsJSON != null ? toolCallArgsJSON : undefined,
      };
    }, {}),
  };
};

/**
 * {@link getToolCallMessageAttributes} wrapped in {@link withSafety} which will return null if any error is thrown
 */
const safelyGetToolCallMessageAttributes = withSafety({
  fn: getToolCallMessageAttributes,
  onError: onErrorCallback("tool call"),
});

/**
 * Gets the OpenInference metadata attributes from ai.telemetry.metadata.*
 * Both vercel and OpenInference attach metadata attributes to spans in a flat structure
 * @example Vercel: ai.telemetry.metadata.<metadataKey>
 * @example OpenInference: metadata.<metadataKey>
 * @param attributes the initial attributes of the span
 * @returns the OpenInference metadata attributes
 */
const getMetadataAttributes = (attributes: Attributes) => {
  const metadataAttributeKeys = Object.keys(attributes)
    .filter((key) => key.startsWith(VercelAISemanticConventions.METADATA))
    .map((key) => ({ key: key.split(".")[3], value: attributes[key] }));
  if (metadataAttributeKeys.length === 0) {
    return null;
  }
  return metadataAttributeKeys.reduce((acc, { key, value }) => {
    return key != null
      ? {
          ...acc,
          [`${SemanticConventions.METADATA}.${key}`]: value,
        }
      : acc;
  }, {});
};

/**
 * {@link getMetadataAttributes} wrapped in {@link withSafety} which will return null if any error is thrown
 */
const safelyGetMetadataAttributes = withSafety({
  fn: getMetadataAttributes,
  onError: onErrorCallback("metadata"),
});

/**
 * Gets the tool call span attributes for TOOL spans
 * @param attributes the span attributes
 * @param spanKind the span kind
 * @returns the OpenInference tool attributes
 */
const getToolCallSpanAttributes = (
  attributes: Attributes,
  spanKind: OpenInferenceSpanKind | string | undefined,
): Attributes | null => {
  if (spanKind !== OpenInferenceSpanKind.TOOL) {
    return null;
  }

  const result: Attributes = {};

  const toolCallId = attributes[VercelAISemanticConventions.TOOL_CALL_ID];
  if (typeof toolCallId === "string") {
    result[SemanticConventions.TOOL_CALL_ID] = toolCallId;
  }

  const toolCallName = attributes[VercelAISemanticConventions.TOOL_CALL_NAME];
  if (typeof toolCallName === "string") {
    result[SemanticConventions.TOOL_NAME] = toolCallName;
  }

  const toolCallArgs = attributes[VercelAISemanticConventions.TOOL_CALL_ARGS];
  if (toolCallArgs != null) {
    result[SemanticConventions.TOOL_PARAMETERS] = toolCallArgs;
    // For tool spans, also set input value
    result[SemanticConventions.INPUT_VALUE] = toolCallArgs;
    result[SemanticConventions.INPUT_MIME_TYPE] =
      getMimeTypeFromValue(toolCallArgs);
  }

  const toolCallResult =
    attributes[VercelAISemanticConventions.TOOL_CALL_RESULT];
  if (toolCallResult != null) {
    result[SemanticConventions.OUTPUT_VALUE] = toolCallResult;
    result[SemanticConventions.OUTPUT_MIME_TYPE] =
      getMimeTypeFromValue(toolCallResult);
  }

  return Object.keys(result).length > 0 ? result : null;
};

/**
 * {@link getToolCallSpanAttributes} wrapped in {@link withSafety} which will return null if any error is thrown
 */
const safelyGetToolCallSpanAttributes = withSafety({
  fn: getToolCallSpanAttributes,
  onError: onErrorCallback("tool call span"),
});

/**
 * Gets streaming performance metrics and stores them as metadata
 * These are Vercel-specific metrics not part of OpenInference spec
 * @param attributes the span attributes
 * @returns metadata attributes for streaming metrics
 */
const getStreamingMetrics = (attributes: Attributes): Attributes | null => {
  const result: Attributes = {};

  const msToFirstChunk =
    attributes[VercelAISemanticConventions.RESPONSE_MS_TO_FIRST_CHUNK];
  if (typeof msToFirstChunk === "number") {
    result[`${SemanticConventions.METADATA}.ai.response.msToFirstChunk`] =
      msToFirstChunk;
  }

  const msToFinish =
    attributes[VercelAISemanticConventions.RESPONSE_MS_TO_FINISH];
  if (typeof msToFinish === "number") {
    result[`${SemanticConventions.METADATA}.ai.response.msToFinish`] =
      msToFinish;
  }

  const avgTokensPerSec =
    attributes[
      VercelAISemanticConventions.RESPONSE_AVG_OUTPUT_TOKENS_PER_SECOND
    ];
  if (typeof avgTokensPerSec === "number") {
    result[
      `${SemanticConventions.METADATA}.ai.response.avgOutputTokensPerSecond`
    ] = avgTokensPerSec;
  }

  return Object.keys(result).length > 0 ? result : null;
};

/**
 * {@link getStreamingMetrics} wrapped in {@link withSafety} which will return null if any error is thrown
 */
const safelyGetStreamingMetrics = withSafety({
  fn: getStreamingMetrics,
  onError: onErrorCallback("streaming metrics"),
});

/**
 * Gets input/output value attributes from Vercel-specific attributes
 * @param attributes the span attributes
 * @param spanKind the span kind
 * @returns the OpenInference IO attributes
 */
const getVercelIOAttributes = (
  attributes: Attributes,
  spanKind: OpenInferenceSpanKind | string | undefined,
): Attributes | null => {
  // Tool spans have their own IO handling
  if (spanKind === OpenInferenceSpanKind.TOOL) {
    return null;
  }

  const result: Attributes = {};

  // Input from ai.prompt (simple prompt string)
  const prompt = attributes[VercelAISemanticConventions.PROMPT];
  if (typeof prompt === "string") {
    const ioAttrs = safelyGetIOValueAttributes({
      attributeValue: prompt,
      OpenInferenceSemanticConventionKey: SemanticConventions.INPUT_VALUE,
    });
    if (ioAttrs) {
      Object.assign(result, ioAttrs);
    }
  }

  // Output from ai.response.text
  const responseText = attributes[VercelAISemanticConventions.RESPONSE_TEXT];
  if (typeof responseText === "string") {
    const ioAttrs = safelyGetIOValueAttributes({
      attributeValue: responseText,
      OpenInferenceSemanticConventionKey: SemanticConventions.OUTPUT_VALUE,
    });
    if (ioAttrs) {
      Object.assign(result, ioAttrs);
    }
  }

  // Output from ai.response.object (for generateObject)
  const responseObject =
    attributes[VercelAISemanticConventions.RESPONSE_OBJECT];
  if (
    typeof responseObject === "string" &&
    !result[SemanticConventions.OUTPUT_VALUE]
  ) {
    const ioAttrs = safelyGetIOValueAttributes({
      attributeValue: responseObject,
      OpenInferenceSemanticConventionKey: SemanticConventions.OUTPUT_VALUE,
    });
    if (ioAttrs) {
      Object.assign(result, ioAttrs);
    }
  }

  return Object.keys(result).length > 0 ? result : null;
};

/**
 * {@link getVercelIOAttributes} wrapped in {@link withSafety} which will return null if any error is thrown
 */
const safelyGetVercelIOAttributes = withSafety({
  fn: getVercelIOAttributes,
  onError: onErrorCallback("Vercel IO"),
});

/**
 * Gets model name from Vercel attributes when gen_ai.* attributes are not present
 * @param attributes the span attributes
 * @param spanKind the span kind
 * @returns the model name attribute
 */
const getModelNameAttribute = (
  attributes: Attributes,
  spanKind: OpenInferenceSpanKind | string | undefined,
): Attributes | null => {
  const modelId = attributes[VercelAISemanticConventions.MODEL_ID];
  if (typeof modelId !== "string") {
    return null;
  }

  const modelSemanticConvention =
    spanKind === OpenInferenceSpanKind.EMBEDDING
      ? SemanticConventions.EMBEDDING_MODEL_NAME
      : SemanticConventions.LLM_MODEL_NAME;

  return {
    [modelSemanticConvention]: modelId,
  };
};

/**
 * {@link getModelNameAttribute} wrapped in {@link withSafety} which will return null if any error is thrown
 */
const safelyGetModelNameAttribute = withSafety({
  fn: getModelNameAttribute,
  onError: onErrorCallback("model name"),
});

/**
 * Gets token count attributes from Vercel ai.usage.* attributes when gen_ai.* are not present
 * @param attributes the span attributes
 * @param spanKind the span kind
 * @returns the token count attributes
 */
const getTokenCountAttributes = (
  attributes: Attributes,
  spanKind: OpenInferenceSpanKind | string | undefined,
): Attributes | null => {
  // Only process token counts for LLM spans
  if (spanKind !== OpenInferenceSpanKind.LLM) {
    return null;
  }

  const result: Attributes = {};

  // Prompt/input tokens
  const promptTokens =
    attributes[VercelAISemanticConventions.TOKEN_COUNT_PROMPT];
  const inputTokens = attributes[VercelAISemanticConventions.TOKEN_COUNT_INPUT];
  const promptCount = promptTokens ?? inputTokens;
  if (typeof promptCount === "number") {
    result[SemanticConventions.LLM_TOKEN_COUNT_PROMPT] = promptCount;
  }

  // Completion/output tokens
  const completionTokens =
    attributes[VercelAISemanticConventions.TOKEN_COUNT_COMPLETION];
  const outputTokens =
    attributes[VercelAISemanticConventions.TOKEN_COUNT_OUTPUT];
  const completionCount = completionTokens ?? outputTokens;
  if (typeof completionCount === "number") {
    result[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION] = completionCount;
  }

  return Object.keys(result).length > 0 ? result : null;
};

/**
 * {@link getTokenCountAttributes} wrapped in {@link withSafety} which will return null if any error is thrown
 */
const safelyGetTokenCountAttributes = withSafety({
  fn: getTokenCountAttributes,
  onError: onErrorCallback("token counts"),
});

/**
 * Gets Vercel-specific attributes that are not covered by gen_ai.* conventions
 * @param attributes - The span attributes
 * @param spanKind - The OpenInference span kind
 * @returns Vercel-specific OpenInference attributes
 */
const getVercelSpecificAttributes = (
  attributes: Attributes,
  spanKind: OpenInferenceSpanKind | string | undefined,
): Attributes => {
  return {
    // Embeddings (only for EMBEDDING spans)
    ...safelyGetEmbeddingAttributes(attributes, spanKind),

    // Tool call spans (only for TOOL spans)
    ...safelyGetToolCallSpanAttributes(attributes, spanKind),

    // Input/Output values from ai.response.* and ai.prompt
    ...safelyGetVercelIOAttributes(attributes, spanKind),

    // Metadata from ai.telemetry.metadata.*
    ...safelyGetMetadataAttributes(attributes),

    // Output messages from ai.response.toolCalls
    ...safelyGetToolCallMessageAttributes(
      attributes[VercelAISemanticConventions.RESPONSE_TOOL_CALLS],
    ),

    // Input messages from ai.prompt.messages
    ...safelyGetInputMessageAttributes(
      attributes[VercelAISemanticConventions.PROMPT_MESSAGES],
    ),

    // Streaming performance metrics (store as metadata)
    ...safelyGetStreamingMetrics(attributes),

    // Model name from ai.model.id (fallback if gen_ai.* not present)
    ...safelyGetModelNameAttribute(attributes, spanKind),

    // Invocation parameters from ai.settings.* (fallback if gen_ai.* not present)
    ...safelyGetInvocationParamAttributes(attributes),

    // Token counts from ai.usage.* (fallback if gen_ai.* not present)
    ...safelyGetTokenCountAttributes(attributes, spanKind),
  };
};

/**
 * Checks if the span has any gen_ai.* attributes
 * @param attributes - The span attributes
 * @returns true if the span has gen_ai attributes
 */
const hasGenAIAttributes = (attributes: Attributes): boolean => {
  return Object.keys(attributes).some((key) => key.startsWith("gen_ai."));
};

/**
 * Convert Vercel AI SDK v6 span attributes to OpenInference attributes.
 *
 * Strategy:
 * 1. First apply openinference-genai to handle standard gen_ai.* attributes
 * 2. Then apply Vercel-specific mappings for ai.* attributes
 * 3. Merge results, with gen_ai mappings taking precedence where overlap exists
 *
 * @param attributes - The initial attributes of the span
 * @returns The OpenInference attributes associated with the span
 */
const getOpenInferenceAttributes = (attributes: Attributes): Attributes => {
  // Step 1: Convert gen_ai.* attributes using openinference-genai
  // Only apply if there are actual gen_ai.* attributes
  const hasGenAI = hasGenAIAttributes(attributes);
  const genAIAttributes = hasGenAI
    ? (convertGenAISpanAttributesToOpenInferenceSpanAttributes(attributes) ??
      {})
    : {};

  // Step 2: Determine span kind from operation.name (Vercel-specific, more precise)
  const spanKind = safelyGetOISpanKindFromAttributes(attributes) ?? undefined;

  // Step 3: Get Vercel-specific attributes not covered by gen_ai.*
  const vercelSpecificAttributes = getVercelSpecificAttributes(
    attributes,
    spanKind,
  );

  // Step 4: Determine final span kind
  // Use Vercel's operation.name-based span kind as it's more specific
  // Only fall back to genai span kind if we have gen_ai attributes
  const finalSpanKind =
    spanKind ??
    (hasGenAI
      ? genAIAttributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]
      : undefined);

  // Step 5: Merge with gen_ai attributes taking precedence for overlapping keys
  return {
    ...vercelSpecificAttributes,
    ...genAIAttributes, // gen_ai takes precedence for model name, tokens, etc.
    [SemanticConventions.OPENINFERENCE_SPAN_KIND]: finalSpanKind,
  };
};

/**
 * {@link getOpenInferenceAttributes} wrapped in {@link withSafety} which will return null if any error is thrown
 */
export const safelyGetOpenInferenceAttributes = withSafety({
  fn: getOpenInferenceAttributes,
  onError: onErrorCallback(""),
});

export const isOpenInferenceSpan = (span: ReadableSpan) => {
  const maybeOpenInferenceSpanKind =
    span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND];
  return typeof maybeOpenInferenceSpanKind === "string";
};

/**
 * Determines whether a span should be exported based on configuration and the spans attributes.
 * @param params - The parameters object
 * @param params.span - The span to check for export eligibility
 * @param params.spanFilter - A filter to apply to a span before exporting. If it returns true for a given span, the span will be exported
 * @returns true if the span should be exported, false otherwise.
 */
export const shouldExportSpan = ({
  span,
  spanFilter,
}: {
  span: ReadableSpan;
  spanFilter?: SpanFilter;
}): boolean => {
  if (spanFilter == null) {
    return true;
  }
  return spanFilter(span);
};

/**
 * Adds OpenInference attributes to a span based on the span's existing attributes.
 * @param span - The span to add OpenInference attributes to.
 */
export const addOpenInferenceAttributesToSpan = (span: ReadableSpan): void => {
  const newAttributes = {
    ...safelyGetOpenInferenceAttributes(span.attributes),
  };

  // newer versions of opentelemetry will not allow you to reassign
  // the attributes object, so you must edit it by keyname instead
  Object.entries(newAttributes).forEach(([key, value]) => {
    span.attributes[key] = value as AttributeValue;
  });
};
