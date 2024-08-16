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
  ReadWriteSpan,
} from "../types";
import { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  assertUnreachable,
  isArrayOfObjects,
  isStringArray,
} from "../utils/typeUtils";
import { isAttributeValue } from "@opentelemetry/core";

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
export const getOISpanKindFromAttributes = (
  attributes: Attributes,
): OpenInferenceSpanKind | null => {
  const maybeOperationName = attributes["operation.name"];
  if (maybeOperationName == null || typeof maybeOperationName !== "string") {
    return null;
  }
  const maybeFunctionName =
    getVercelFunctionNameFromOperationName(maybeOperationName);
  if (maybeFunctionName == null) {
    return null;
  }
  return VercelSDKFunctionNameToSpanKindMap.get(maybeFunctionName) ?? null;
};

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
      JSON.stringify(settingAttributes),
  };
};

/**
 * Determines whether the value is a valid JSON string
 * @param value the value to check
 * @returns whether the value is a valid JSON string
 */
const isValidJsonString = (value?: AttributeValue) => {
  if (typeof value !== "string") {
    return false;
  }

  try {
    const parsed = JSON.parse(value);
    return typeof parsed === "object" && parsed !== null;
  } catch (e) {
    return false;
  }
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
 * Formats an embedding attribute value (i.e., embedding text or vector) into the expected format
 * Vercel embedding vector attributes are stringified arrays, however, the OpenInference spec expects them to be un-stringified arrays
 * @param value the value to format (either embedding text or vector)
 * @returns the formatted value or the original value if it is not a string or cannot be parsed
 */
const formatEmbeddingValue = (value: AttributeValue) => {
  if (typeof value !== "string") {
    return value;
  }
  try {
    const parsedValue = JSON.parse(value);
    if (isAttributeValue(parsedValue)) {
      return parsedValue;
    }
  } catch (e) {
    return value;
  }
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
 * Gets the input_messages OpenInference attributes
 * @param promptMessages the attribute value of the Vercel prompt messages
 * @returns input_messages attributes
 */
const getInputMessageAttributes = (promptMessages?: AttributeValue) => {
  if (typeof promptMessages !== "string") {
    return null;
  }

  const messages = JSON.parse(promptMessages);

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
 * Gets the output_messages tool_call OpenInference attributes
 * @param toolCalls the attribute value of the Vercel result.toolCalls
 * @returns output_messages tool_call attributes
 */
const getToolCallMessageAttributes = (toolCalls?: AttributeValue) => {
  if (typeof toolCalls !== "string") {
    return null;
  }

  const parsedToolCalls = JSON.parse(toolCalls);

  if (!isArrayOfObjects(parsedToolCalls)) {
    return null;
  }

  const OUTPUT_MESSAGE_PREFIX = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0`;
  return {
    [`${OUTPUT_MESSAGE_PREFIX}.${SemanticConventions.MESSAGE_ROLE}`]:
      "assistant",
    ...parsedToolCalls.reduce((acc: Attributes, toolCall, index) => {
      const TOOL_CALL_PREFIX = `${OUTPUT_MESSAGE_PREFIX}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${index}`;
      return {
        ...acc,
        [`${TOOL_CALL_PREFIX}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`]:
          isAttributeValue(toolCall.toolName) ? toolCall.toolName : undefined,
        [`${TOOL_CALL_PREFIX}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`]:
          toolCall.args != null ? JSON.stringify(toolCall.args) : undefined,
      };
    }, {}),
  };
};

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
    .map((key) => ({ key: key.split(".")[2], value: attributes[key] }));
  if (metadataAttributeKeys.length === 0) {
    return null;
  }
  return metadataAttributeKeys.reduce(
    (acc, { key, value }) => ({
      ...acc,
      [`${SemanticConventions.METADATA}.${key}`]: value,
    }),
    {},
  );
};

/**
 * Gets the OpenInference attributes associated with the span from the initial attributes
 * @param attributesWithSpanKind the initial attributes of the span and the OpenInference span kind
 * @param attributesWithSpanKind.initialAttributes the initial attributes of the span
 * @param attributesWithSpanKind.spanKind the OpenInference span kind
 * @returns The OpenInference attributes associated with the span
 */
export const getOpenInferenceAttributes = ({
  initialAttributes,
  spanKind,
}: {
  initialAttributes: Attributes;
  spanKind: OpenInferenceSpanKind;
}): Attributes => {
  return VercelSemanticConventionsList.reduce(
    (openInferenceAttributes: Attributes, convention) => {
      /**
       *  Both settings and metadata are not full attribute paths but prefixes
       * @example ai.settings.<paramName> or ai.metadata.<metadataKey>
       */
      if (
        !(convention in initialAttributes) &&
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
            ...getMetadataAttributes(initialAttributes),
          };
        case VercelSemanticConventions.TOKEN_COUNT_COMPLETION:
        case VercelSemanticConventions.TOKEN_COUNT_PROMPT:
        case VercelSemanticConventions.TOOL_CALL_NAME:
        case VercelSemanticConventions.TOOL_CALL_ARGS:
          return {
            ...openInferenceAttributes,
            [openInferenceKey]: initialAttributes[convention],
          };
        case VercelSemanticConventions.MODEL_ID: {
          const modelSemanticConvention =
            spanKind === OpenInferenceSpanKind.EMBEDDING
              ? SemanticConventions.EMBEDDING_MODEL_NAME
              : SemanticConventions.LLM_MODEL_NAME;
          return {
            ...openInferenceAttributes,
            [modelSemanticConvention]: initialAttributes[convention],
          };
        }
        case VercelSemanticConventions.SETTINGS:
          return {
            ...openInferenceAttributes,
            ...getInvocationParamAttributes(initialAttributes),
          };
        case VercelSemanticConventions.PROMPT:
        case VercelSemanticConventions.RESULT_OBJECT:
        case VercelSemanticConventions.RESULT_TEXT: {
          return {
            ...openInferenceAttributes,
            ...getIOValueAttributes({
              attributeValue: initialAttributes[convention],
              openInferenceSemanticConvention:
                openInferenceKey as OpenInferenceIOConvention,
            }),
          };
        }
        case VercelSemanticConventions.RESULT_TOOL_CALLS:
          return {
            ...openInferenceAttributes,
            ...getToolCallMessageAttributes(initialAttributes[convention]),
          };
        case VercelSemanticConventions.PROMPT_MESSAGES:
          return {
            ...openInferenceAttributes,
            ...getInputMessageAttributes(initialAttributes[convention]),
          };
          break;
        case VercelSemanticConventions.EMBEDDING_TEXT:
        case VercelSemanticConventions.EMBEDDING_TEXTS:
        case VercelSemanticConventions.EMBEDDING_VECTOR:
        case VercelSemanticConventions.EMBEDDING_VECTORS:
          return {
            ...openInferenceAttributes,
            ...getEmbeddingAttributes({
              attributeValue: initialAttributes[convention],
              openInferenceSemanticConvention: openInferenceKey,
            }),
          };
        default:
          return assertUnreachable(convention);
      }
    },
    {},
  );
};

/**
 * Makes a copy of the spans passed to the exporter
 * Note: While these are typed as ReadableSpans, they are actually still Span instances.
 * As such, making a copy here does not deeply copy all of the methods on the span, however since this is happening at the exporter level, it is safe to assume that the spans will be read-only from this point on and will not need access to setter methods (e.g., addEvent, addLink, etc.)
 * @param spans the spans passed to the exporter
 * @returns a copy of the spans passed to the exporter with the ability to set attributes
 */
const copySpans = (spans: ReadableSpan[]): ReadWriteSpan[] => {
  return spans.map((span) => {
    // spanContext is a getter so it needs to be explicitly copied
    const mutableSpan: ReadWriteSpan = {
      ...span,
      spanContext: () => span.spanContext(),
    };
    return mutableSpan;
  });
};

/**
 * Takes the spans passed to the exporter and adds OpenInference attributes to them
 * @param spans the spans passed to the exporter
 * @returns spans with OpenInference attributes added
 */
export const addOpenInferenceAttributesToSpans = (
  spans: ReadableSpan[],
): ReadWriteSpan[] => {
  const mutableSpans = copySpans(spans);
  return mutableSpans.map((span) => {
    const initialAttributes = span.attributes;
    const spanKind = getOISpanKindFromAttributes(initialAttributes);

    //  Only modify spans that have a corresponding OpenInferenceSpanKind
    if (spanKind == null) {
      return span;
    }
    span.attributes = {
      ...span.attributes,
      ...getOpenInferenceAttributes({ initialAttributes, spanKind }),
      [SemanticConventions.OPENINFERENCE_SPAN_KIND]: spanKind,
    };
    return span;
  });
};
