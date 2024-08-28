import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { Attributes, AttributeValue, diag } from "@opentelemetry/api";
import {
  VercelSDKFunctionNameToSpanKindMap,
  AISemConvToOISemConvMap,
} from "./constants";
import {
  AISemanticConventions,
  AISemanticConventionsList,
} from "./AISemanticConventions";
import {
  OpenInferenceIOConventionKey,
  OpenInferenceSemanticConventionKey,
  ReadWriteSpan,
  SpanFilter,
} from "./types";
import {
  assertUnreachable,
  isArrayOfObjects,
  isStringArray,
} from "./typeUtils";
import { isAttributeValue } from "@opentelemetry/core";
import {
  safelyJSONParse,
  safelyJSONStringify,
  withSafety,
} from "@arizeai/openinference-core";
import { ReadableSpan } from "@opentelemetry/sdk-trace-base";

const onErrorCallback = (attributeType: string) => (error: unknown) => {
  diag.warn(
    `Unable to get OpenInference ${attributeType} attributes from AI attributes falling back to null: ${error}`,
  );
};

/**
 *
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
 * Gets the OpenInference span kind that corresponds to the Vercel operation name
 * @param attributes the attributes of the span
 * @returns the OpenInference span kind associated with the attributes or null if not found
 */
const getOISpanKindFromAttributes = (
  attributes: Attributes,
): OpenInferenceSpanKind | undefined => {
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
 * @param attributes the initial attributes of the span
 * @returns the OpenInference attributes associated with the invocation parameters
 */
const getInvocationParamAttributes = (attributes: Attributes) => {
  const settingAttributeKeys = Object.keys(attributes).filter((key) =>
    key.startsWith(AISemanticConventions.SETTINGS),
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
 * Takes the Vercel embedding attribute value and the corresponding OpenInference attribute key and returns the OpenInference attributes associated with the embedding
 * The Vercel embedding attribute value can be a string or an array of strings
 * @param object the attribute value and the OpenInferenceSemanticConventionKey (either EMBEDDING_TEXT or EMBEDDING_VECTOR)
 * @returns the OpenInference attributes associated with the embedding
 */
const getEmbeddingAttributes = ({
  attributeValue,
  OpenInferenceSemanticConventionKey,
}: {
  attributeValue?: AttributeValue;
  OpenInferenceSemanticConventionKey: OpenInferenceSemanticConventionKey;
}) => {
  const EMBEDDING_PREFIX = SemanticConventions.EMBEDDING_EMBEDDINGS;

  if (typeof attributeValue === "string") {
    return {
      [`${EMBEDDING_PREFIX}.0.${OpenInferenceSemanticConventionKey}`]:
        formatEmbeddingValue(attributeValue),
    };
  }
  if (isStringArray(attributeValue)) {
    return attributeValue.reduce((acc: Attributes, embeddingValue, index) => {
      acc[
        `${EMBEDDING_PREFIX}.${index}.${OpenInferenceSemanticConventionKey}`
      ] = formatEmbeddingValue(embeddingValue);
      return acc;
    }, {});
  }
  return null;
};

/**
 * {@link getEmbeddingAttributes} wrapped in {@link withSafety} which will return null if any error is thrown
 */
const safelyGetEmbeddingAttributes = withSafety({
  fn: getEmbeddingAttributes,
  onError: onErrorCallback("embedding"),
});

/**
 * Gets the input_messages OpenInference attributes
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

  return messages.reduce((acc: Attributes, message, index) => {
    const MESSAGE_PREFIX = `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}`;
    if (isArrayOfObjects(message.content)) {
      const messageAttributes = message.content.reduce(
        (acc: Attributes, content, contentIndex) => {
          const CONTENTS_PREFIX = `${MESSAGE_PREFIX}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}`;
          return {
            ...acc,
            [`${CONTENTS_PREFIX}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`]:
              typeof content.type === "string" ? content.type : undefined,
            [`${CONTENTS_PREFIX}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`]:
              typeof content.text === "string" ? content.text : undefined,
            [`${CONTENTS_PREFIX}.${SemanticConventions.MESSAGE_CONTENT_IMAGE}`]:
              typeof content.image === "string" ? content.image : undefined,
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
    acc[
      `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_ROLE}`
    ] = typeof message.role === "string" ? message.role : undefined;
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
 * @param toolCalls the attribute value of the Vercel result.toolCalls
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
      const toolCallArgsJSON = safelyJSONStringify(toolCall.args);
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
 * Gets the OpenInference metadata attributes
 * Both vercel and OpenInference attach metadata attributes to spans in a flat structure
 * @example Vercel: ai.telemetry.metadata.<metadataKey>
 * @example OpenInference: metadata.<metadataKey>
 * @param attributes the initial attributes of the span
 * @returns the OpenInference metadata attributes
 */
const getMetadataAttributes = (attributes: Attributes) => {
  const metadataAttributeKeys = Object.keys(attributes)
    .filter((key) => key.startsWith(AISemanticConventions.METADATA))
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
 * Gets the OpenInference attributes associated with the span from the initial attributes
 * @param attributesWithSpanKind the initial attributes of the span and the OpenInference span kind
 * @param attributesWithSpanKind.attributes the initial attributes of the span
 * @param attributesWithSpanKind.spanKind the OpenInference span kind
 * @returns The OpenInference attributes associated with the span
 */
const getOpenInferenceAttributes = (attributes: Attributes): Attributes => {
  const spanKind = safelyGetOISpanKindFromAttributes(attributes);
  const openInferenceAttributes = {
    [SemanticConventions.OPENINFERENCE_SPAN_KIND]: spanKind ?? undefined,
  };
  return AISemanticConventionsList.reduce(
    (openInferenceAttributes: Attributes, convention) => {
      /**
       *  Both settings and metadata are not full attribute paths but prefixes
       * @example ai.settings.<paramName> or ai.metadata.<metadataKey>
       */
      if (
        !(convention in attributes) &&
        convention !== AISemanticConventions.SETTINGS &&
        convention !== AISemanticConventions.METADATA
      ) {
        return openInferenceAttributes;
      }

      const openInferenceKey = AISemConvToOISemConvMap[convention];

      switch (convention) {
        case AISemanticConventions.METADATA:
          return {
            ...openInferenceAttributes,
            ...safelyGetMetadataAttributes(attributes),
          };
        case AISemanticConventions.TOKEN_COUNT_COMPLETION:
        case AISemanticConventions.TOKEN_COUNT_PROMPT:
          // Do not capture token counts for non LLM spans to avoid double token counts
          if (spanKind !== OpenInferenceSpanKind.LLM) {
            return openInferenceAttributes;
          }
          return {
            ...openInferenceAttributes,
            [openInferenceKey]: attributes[convention],
          };
        case AISemanticConventions.TOOL_CALL_NAME:
          return {
            ...openInferenceAttributes,
            [openInferenceKey]: attributes[convention],
          };
        case AISemanticConventions.TOOL_CALL_ARGS: {
          let argsAttributes = {
            [openInferenceKey]: attributes[convention],
          };
          // For tool spans, capture the arguments as input value
          if (spanKind === OpenInferenceSpanKind.TOOL) {
            argsAttributes = {
              ...argsAttributes,
              [SemanticConventions.INPUT_VALUE]: attributes[convention],
              [SemanticConventions.INPUT_MIME_TYPE]: getMimeTypeFromValue(
                attributes[convention],
              ),
            };
          }
          return {
            ...openInferenceAttributes,
            ...argsAttributes,
          };
        }
        case AISemanticConventions.TOOL_CALL_RESULT:
          // For tool spans, capture the result as output value, for non tool spans ignore
          if (spanKind !== OpenInferenceSpanKind.TOOL) {
            return openInferenceAttributes;
          }
          return {
            ...openInferenceAttributes,
            [openInferenceKey]: attributes[convention],
            [SemanticConventions.OUTPUT_MIME_TYPE]: getMimeTypeFromValue(
              attributes[convention],
            ),
          };
        case AISemanticConventions.MODEL_ID: {
          const modelSemanticConvention =
            spanKind === OpenInferenceSpanKind.EMBEDDING
              ? SemanticConventions.EMBEDDING_MODEL_NAME
              : SemanticConventions.LLM_MODEL_NAME;
          return {
            ...openInferenceAttributes,
            [modelSemanticConvention]: attributes[convention],
          };
        }
        case AISemanticConventions.SETTINGS:
          return {
            ...openInferenceAttributes,
            ...safelyGetInvocationParamAttributes(attributes),
          };
        case AISemanticConventions.PROMPT:
        case AISemanticConventions.RESULT_OBJECT:
        case AISemanticConventions.RESULT_TEXT: {
          return {
            ...openInferenceAttributes,
            ...safelyGetIOValueAttributes({
              attributeValue: attributes[convention],
              OpenInferenceSemanticConventionKey:
                openInferenceKey as OpenInferenceIOConventionKey,
            }),
          };
        }
        case AISemanticConventions.RESULT_TOOL_CALLS:
          return {
            ...openInferenceAttributes,
            ...safelyGetToolCallMessageAttributes(attributes[convention]),
          };
        case AISemanticConventions.PROMPT_MESSAGES:
          return {
            ...openInferenceAttributes,
            ...safelyGetInputMessageAttributes(attributes[convention]),
          };
          break;
        case AISemanticConventions.EMBEDDING_TEXT:
        case AISemanticConventions.EMBEDDING_TEXTS:
        case AISemanticConventions.EMBEDDING_VECTOR:
        case AISemanticConventions.EMBEDDING_VECTORS:
          return {
            ...openInferenceAttributes,
            ...safelyGetEmbeddingAttributes({
              attributeValue: attributes[convention],
              OpenInferenceSemanticConventionKey: openInferenceKey,
            }),
          };
        default:
          return assertUnreachable(convention);
      }
    },
    openInferenceAttributes,
  );
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
 * @param span the spn to check for export eligibility.
 * @param spanFilters a list of filters to apply to a span before exporting. If at least one filter returns true for a given span, the span will be exported.
 * @returns true if the span should be exported, false otherwise.
 */
export const shouldExportSpan = ({
  span,
  spanFilters,
}: {
  span: ReadableSpan;
  spanFilters?: SpanFilter[];
}): boolean => {
  if (spanFilters == null) {
    return true;
  }
  return spanFilters.some((filter) => filter(span));
};

/**
 * Adds OpenInference attributes to a span based on the span's existing attributes.
 * @param span - The span to add OpenInference attributes to.
 */
export const addOpenInferenceAttributesToSpan = (span: ReadableSpan): void => {
  const attributes = { ...span.attributes };

  (span as ReadWriteSpan).attributes = {
    ...span.attributes,
    ...safelyGetOpenInferenceAttributes(attributes),
  };
};
