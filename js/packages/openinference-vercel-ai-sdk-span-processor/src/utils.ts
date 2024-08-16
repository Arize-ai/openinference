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
} from "./types";
import { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { isArrayOfObjects, isStringArray } from "./typeUtils";

const assertUnreachable = (x: never): never => {
  throw new Error("Unexpected value: " + x);
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
  const maybeSDKPrefix = operationName.split(" ")[0];
  if (maybeSDKPrefix == null) {
    return;
  }
  return maybeSDKPrefix.split(".")[2];
};

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
 * Takes the attributes from the span and accumulates the attributes that are prefixed with "ai.settings"
 * @param attributes
 * @returns
 */
const getInvocationParamAttributes = (attributes: Attributes) => {
  const settingAttributes = Object.keys(attributes)
    .filter((key) => key.startsWith(VercelSemanticConventions.SETTINGS))
    .reduce((acc, key) => {
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

const getMimeTypeFromValue = (value?: AttributeValue) => {
  if (isValidJsonString(value)) {
    return MimeType.JSON;
  }
  return MimeType.TEXT;
};

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
        attributeValue,
    };
  }
  if (isStringArray(attributeValue)) {
    return attributeValue.reduce((acc: Attributes, embedding, index) => {
      acc[`${EMBEDDING_PREFIX}.${index}.${openInferenceSemanticConvention}`] =
        embedding;
      return acc;
    }, {});
  }
  return null;
};

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

export const getOpenInferenceAttributes = (initialAttributes: Attributes) => {
  return VercelSemanticConventionsList.reduce(
    (openInferenceAttributes: Attributes, convention) => {
      if (
        !(convention in initialAttributes) &&
        convention !== VercelSemanticConventions.SETTINGS
      ) {
        return openInferenceAttributes;
      }

      const openInferenceKey = VercelSemConvToOISemConvMap[convention];

      switch (convention) {
        case VercelSemanticConventions.MODEL_ID:
        case VercelSemanticConventions.TOKEN_COUNT_COMPLETION:
        case VercelSemanticConventions.TOKEN_COUNT_PROMPT:
        case VercelSemanticConventions.METADATA:
          openInferenceAttributes[openInferenceKey] =
            initialAttributes[convention];
          break;
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
          break;
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
        case VercelSemanticConventions.TOOL_CALL_NAME:
        case VercelSemanticConventions.TOOL_CALL_ARGS:
          break;
        default:
          return assertUnreachable(convention);
      }
      return openInferenceAttributes;
    },
    {},
  );
};

/**
 * Makes a copy of the spans passed to the exporter and casts them to ReadWriteSpan to allow for attribute setting
 * @param spans
 * @returns a copy of the spans passed to the exporter with the ability to set attributes
 */
export const copySpans = (spans: ReadableSpan[]): ReadWriteSpan[] => {
  return spans.map((span) => {
    // spanContext is a getter so it needs to be explicitly copied
    const mutableSpan = {
      ...span,
      spanContext: span.spanContext,
    };
    // Cast here to ReadWriteSpan which contains attribute setter methods
    // The spans coming into the exporter are typed as ReadableSpans but actually contain the attribute setter methods
    return mutableSpan as ReadWriteSpan;
  });
};
