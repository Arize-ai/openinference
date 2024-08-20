import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { Attributes, AttributeValue } from "@opentelemetry/api";
import {
  VercelSDKFunctionNameToSpanKindMap,
  VercelSemConvToOISemConvMap,
} from "./constants";
import {
  VercelSemanticConventions,
  VercelSemanticConventionsList,
} from "./VercelSemanticConventions";
import {
  OpenInferenceIOConvention,
  OpenInferenceSemanticConvention,
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
const safelyGetOISpanKindFromAttributes = withSafety(
  getOISpanKindFromAttributes,
);

/**
 * Takes the attributes from the span and accumulates the attributes that are prefixed with "ai.settings" to be used as the invocation parameters
 * @param attributes the initial attributes of the span
 * @returns the OpenInference attributes associated with the invocation parameters
 */
const getInvocationParamAttributes = (attributes: Attributes) => {
  const settingAttributeKeys = Object.keys(attributes).filter((key) =>
    key.startsWith(VercelSemanticConventions.SETTINGS),
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
const safelyGetInvocationParamAttributes = withSafety(
  getInvocationParamAttributes,
);

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
 * @param object.openInferenceSemanticConvention the corresponding OpenInference semantic convention
 * @returns the OpenInference attributes associated with the IO value
 */
const getIOValueAttributes = ({
  attributeValue,
  openInferenceSemanticConvention,
}: {
  attributeValue?: AttributeValue;
  openInferenceSemanticConvention: OpenInferenceIOConvention;
}) => {
  const mimeTypeSemanticConvention =
    openInferenceSemanticConvention === SemanticConventions.INPUT_VALUE
      ? SemanticConventions.INPUT_MIME_TYPE
      : SemanticConventions.OUTPUT_MIME_TYPE;
  return {
    [openInferenceSemanticConvention]: attributeValue,
    [mimeTypeSemanticConvention]: getMimeTypeFromValue(attributeValue),
  };
};

/**
 * {@link getIOValueAttributes} wrapped in {@link withSafety} which will return null if any error is thrown
 */
const safelyGetIOValueAttributes = withSafety(getIOValueAttributes);

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
 * @param object the attribute value and the OpenInferenceSemanticConvention (either EMBEDDING_TEXT or EMBEDDING_VECTOR)
 * @returns the OpenInference attributes associated with the embedding
 */
const getEmbeddingAttributes = ({
  attributeValue,
  openInferenceSemanticConvention,
}: {
  attributeValue?: AttributeValue;
  openInferenceSemanticConvention: OpenInferenceSemanticConvention;
}) => {
  const EMBEDDING_PREFIX = SemanticConventions.EMBEDDING_EMBEDDINGS;

  if (typeof attributeValue === "string") {
    return {
      [`${EMBEDDING_PREFIX}.0.${openInferenceSemanticConvention}`]:
        formatEmbeddingValue(attributeValue),
    };
  }
  if (isStringArray(attributeValue)) {
    return attributeValue.reduce((acc: Attributes, embeddingValue, index) => {
      acc[`${EMBEDDING_PREFIX}.${index}.${openInferenceSemanticConvention}`] =
        formatEmbeddingValue(embeddingValue);
      return acc;
    }, {});
  }
  return null;
};

/**
 * {@link getEmbeddingAttributes} wrapped in {@link withSafety} which will return null if any error is thrown
 */
const safelyGetEmbeddingAttributes = withSafety(getEmbeddingAttributes);

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
const safelyGetInputMessageAttributes = withSafety(getInputMessageAttributes);

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
const safelyGetToolCallMessageAttributes = withSafety(
  getToolCallMessageAttributes,
);

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
    .filter((key) => key.startsWith(VercelSemanticConventions.METADATA))
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
const safelyGetMetadataAttributes = withSafety(getMetadataAttributes);

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
  return VercelSemanticConventionsList.reduce(
    (openInferenceAttributes: Attributes, convention) => {
      /**
       *  Both settings and metadata are not full attribute paths but prefixes
       * @example ai.settings.<paramName> or ai.metadata.<metadataKey>
       */
      if (
        !(convention in attributes) &&
        convention !== VercelSemanticConventions.SETTINGS &&
        convention !== VercelSemanticConventions.METADATA
      ) {
        return openInferenceAttributes;
      }

      const openInferenceKey = VercelSemConvToOISemConvMap[convention];

      switch (convention) {
        case VercelSemanticConventions.METADATA:
          return {
            ...openInferenceAttributes,
            ...safelyGetMetadataAttributes(attributes),
          };
        case VercelSemanticConventions.TOKEN_COUNT_COMPLETION:
        case VercelSemanticConventions.TOKEN_COUNT_PROMPT:
        case VercelSemanticConventions.TOOL_CALL_NAME:
        case VercelSemanticConventions.TOOL_CALL_ARGS:
          return {
            ...openInferenceAttributes,
            [openInferenceKey]: attributes[convention],
          };
        case VercelSemanticConventions.MODEL_ID: {
          const modelSemanticConvention =
            spanKind === OpenInferenceSpanKind.EMBEDDING
              ? SemanticConventions.EMBEDDING_MODEL_NAME
              : SemanticConventions.LLM_MODEL_NAME;
          return {
            ...openInferenceAttributes,
            [modelSemanticConvention]: attributes[convention],
          };
        }
        case VercelSemanticConventions.SETTINGS:
          return {
            ...openInferenceAttributes,
            ...safelyGetInvocationParamAttributes(attributes),
          };
        case VercelSemanticConventions.PROMPT:
        case VercelSemanticConventions.RESULT_OBJECT:
        case VercelSemanticConventions.RESULT_TEXT: {
          return {
            ...openInferenceAttributes,
            ...safelyGetIOValueAttributes({
              attributeValue: attributes[convention],
              openInferenceSemanticConvention:
                openInferenceKey as OpenInferenceIOConvention,
            }),
          };
        }
        case VercelSemanticConventions.RESULT_TOOL_CALLS:
          return {
            ...openInferenceAttributes,
            ...safelyGetToolCallMessageAttributes(attributes[convention]),
          };
        case VercelSemanticConventions.PROMPT_MESSAGES:
          return {
            ...openInferenceAttributes,
            ...safelyGetInputMessageAttributes(attributes[convention]),
          };
          break;
        case VercelSemanticConventions.EMBEDDING_TEXT:
        case VercelSemanticConventions.EMBEDDING_TEXTS:
        case VercelSemanticConventions.EMBEDDING_VECTOR:
        case VercelSemanticConventions.EMBEDDING_VECTORS:
          return {
            ...openInferenceAttributes,
            ...safelyGetEmbeddingAttributes({
              attributeValue: attributes[convention],
              openInferenceSemanticConvention: openInferenceKey,
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
export const safelyGetOpenInferenceAttributes = withSafety(
  getOpenInferenceAttributes,
);
