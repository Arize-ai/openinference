import {
  MimeType,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { Attributes } from "@opentelemetry/api";
import { Message, TokenCount, DocumentReference } from "./types";
import { parseSanitizedJson } from "../utils/jsonUtils";
import {
  isObjectWithStringKeys,
  safelyJSONStringify,
} from "@arizeai/openinference-core";
import { isAttributeValue } from "@opentelemetry/core";
import { getStringAttributeValueFromUnknown } from "./attributeExtractionUtils";

/**
 * Utility functions for extracting input and output attributes from a text.
 */

/**
 * Normalize MIME type from text or object.
 * Returns MimeType.JSON for JSON-like strings or objects, otherwise MimeType.TEXT.
 * @param input Input string or object to check.
 * @returns MimeType for the input.
 */
function normalizeMimeType(input: unknown): MimeType {
  if (typeof input === "string") {
    const trimmed = input.trim();
    const parsed = parseSanitizedJson(trimmed);
    if (typeof parsed === "object" || Array.isArray(parsed)) {
      return MimeType.JSON;
    }
    return MimeType.TEXT;
  }
  if (typeof input === "object" || Array.isArray(input)) {
    return MimeType.JSON;
  }
  return MimeType.TEXT;
}

/**
 * Extract input attributes from a text or object.
 * Adds value and MIME type to the attributes.
 * @param input Text or object to extract attributes from.
 * @returns Attributes an object with input value and MIME type.
 */
export function getInputAttributes(input: unknown): Attributes {
  const attributes: Attributes = {};
  if (input == null) {
    return attributes;
  }
  const mimeType = normalizeMimeType(input);
  attributes[SemanticConventions.INPUT_MIME_TYPE] = mimeType;
  if (typeof input === "string") {
    attributes[SemanticConventions.INPUT_VALUE] = input;
  } else {
    attributes[SemanticConventions.INPUT_VALUE] =
      safelyJSONStringify(input) ?? undefined;
  }
  return attributes;
}

/**
 * Extracts output attributes from a given text or object.
 * @param value The output value to extract attributes from.
 * @param options Optional configuration for MIME type.
 * @returns Attributes an object with output value and MIME type.
 */
export function getOutputAttributes(value: unknown): Attributes {
  const attributes: Attributes = {};
  const mimeType = normalizeMimeType(value);

  attributes[SemanticConventions.OUTPUT_VALUE] = isAttributeValue(value)
    ? value
    : (safelyJSONStringify(value) ?? undefined);
  attributes[SemanticConventions.OUTPUT_MIME_TYPE] = mimeType;
  return attributes;
}

/**
 * Extracts system attributes for LLM tracing.
 * Accepts either an object with a 'value' property or a string system name.
 * Returns an Attributes object with the system name in lowercase.
 * @param system - System object or string.
 * @returns Attributes an object for system attributes.
 */
export function getLLMSystemAttributes(system: unknown): Attributes {
  if (isObjectWithStringKeys(system)) {
    const value = getStringAttributeValueFromUnknown(system.value);
    return value ? { [SemanticConventions.LLM_SYSTEM]: value } : {};
  }
  if (typeof system === "string") {
    return { [SemanticConventions.LLM_SYSTEM]: system.toLowerCase() };
  }
  return {};
}

/**
 * Extracts invocation parameter attributes for LLM tracing.
 * Accepts either a string or an object for invocation parameters.
 * Returns an Attributes object with invocation parameterss a string.
 * @param invocationParameters - Invocation parameters as string or object.
 * @returns Attributes object for invocation parameters.
 */
export function getLLMInvocationParameterAttributes(
  invocationParameters: unknown,
): Attributes {
  if (typeof invocationParameters === "string") {
    return {
      [SemanticConventions.LLM_INVOCATION_PARAMETERS]: invocationParameters,
    };
  } else if (
    typeof invocationParameters === "object" &&
    invocationParameters !== null
  ) {
    return {
      [SemanticConventions.LLM_INVOCATION_PARAMETERS]:
        JSON.stringify(invocationParameters),
    };
  }
  return {};
}

/**
 * Extracts input message attributes for LLM tracing.
 * Returns an Attributes object with input message attributes.
 * @param messages - Array of input message objects.
 * @returns Attributes an object for input messages.
 */
export function getLLMInputMessageAttributes(
  messages: Message[] | undefined,
): Attributes {
  return {
    ...llmMessagesAttributes(messages, "input"),
  };
}

/**
 * Extracts model name attributes for LLM tracing.
 * Returns an Attributes object with the model name.
 * @param modelName - Model name as string.
 * @returns Attributes an object for model name.
 */
export function getLLMModelNameAttributes(modelName: unknown): Attributes {
  if (typeof modelName === "string") {
    return { [SemanticConventions.LLM_MODEL_NAME]: modelName };
  }
  return {};
}

/**
 * Extracts output message attributes for LLM tracing.
 * Returns an Attributes object with output message attributes.
 * @param messages - Array of output message objects.
 * @returns Attributes an object for output messages.
 */
export function getLLMOutputMessageAttributes(
  messages: Message[] | undefined,
): Attributes {
  return {
    ...llmMessagesAttributes(messages, "output"),
  };
}

export function getLLMTokenCountAttributes(
  tokenCount: TokenCount | undefined,
): Attributes {
  const attributes: Attributes = {};
  if (tokenCount && typeof tokenCount === "object") {
    if (tokenCount.prompt !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT] =
        tokenCount.prompt;
    }
    if (tokenCount.completion !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION] =
        tokenCount.completion;
    }
    if (tokenCount.total !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL] = tokenCount.total;
    }
  }
  return attributes;
}

// get_llm_tool_attributes
export function getLLMToolAttributes(
  tools: Record<string, unknown>[] | undefined,
): Attributes {
  const attributes: Attributes = {};
  if (!Array.isArray(tools)) {
    return attributes;
  }
  for (let toolIndex = 0; toolIndex < tools.length; toolIndex++) {
    const tool = tools[toolIndex];
    if (typeof tool !== "object" || tool === null) continue;
    const toolJsonSchema = tool["json_schema"];
    if (typeof toolJsonSchema === "string") {
      attributes[
        `${SemanticConventions.LLM_TOOLS}.${toolIndex}.${SemanticConventions.TOOL_JSON_SCHEMA}`
      ] = toolJsonSchema;
    } else if (typeof toolJsonSchema === "object" && toolJsonSchema !== null) {
      attributes[
        `${SemanticConventions.LLM_TOOLS}.${toolIndex}.${SemanticConventions.TOOL_JSON_SCHEMA}`
      ] = JSON.stringify(toolJsonSchema);
    }
  }
  return attributes;
}

export function getLLMAttributes({
  provider,
  system,
  modelName,
  invocationParameters,
  inputMessages,
  outputMessages,
  tokenCount,
  tools,
}: {
  provider?: string;
  system?: string;
  modelName?: string;
  invocationParameters?: string | Record<string, unknown>;
  inputMessages?: Message[];
  outputMessages?: Message[];
  tokenCount?: TokenCount;
  tools?: Record<string, unknown>[];
} = {}): Attributes {
  const providerAttributes = getLLMProviderAttributes(provider);
  const modelNameAttributes = getLLMModelNameAttributes(modelName);
  const systemAttributes = getLLMSystemAttributes(system);
  const invocationParameterAttributes =
    getLLMInvocationParameterAttributes(invocationParameters);
  const inputMessageAttributes = getLLMInputMessageAttributes(inputMessages);
  const outputMessageAttributes = getLLMOutputMessageAttributes(outputMessages);
  const tokenCountAttributes = getLLMTokenCountAttributes(tokenCount);
  const toolAttributes = getLLMToolAttributes(tools);
  return {
    ...providerAttributes,
    ...systemAttributes,
    ...modelNameAttributes,
    ...invocationParameterAttributes,
    ...inputMessageAttributes,
    ...outputMessageAttributes,
    ...tokenCountAttributes,
    ...toolAttributes,
  };
}

/**
 * Extracts tool attributes for tracing.
 * Accepts tool name, description, and parameters (as string or object).
 * Returns an Attributes object with tool name, parameters, and optional description.
 * @param name - The name of the tool.
 * @param description - Optional description of the tool.
 * @param parameters - Tool parameters as string or object.
 * @returns Attributes an object for tool attributes.
 */
export function getToolAttributes({
  name,
  description,
  parameters,
}: {
  name?: string;
  description?: string;
  parameters: string | Record<string, unknown>;
}): Attributes {
  let parametersJson: string;
  if (typeof parameters === "string") {
    parametersJson = parameters;
  } else if (typeof parameters === "object" && parameters !== null) {
    parametersJson = JSON.stringify(parameters);
  } else {
    throw new Error(`Invalid parameters type: ${typeof parameters}`);
  }
  const attributes: Attributes = {
    [SemanticConventions.TOOL_NAME]: name,
    [SemanticConventions.TOOL_PARAMETERS]: parametersJson,
  };
  if (description !== undefined && description !== null) {
    attributes[SemanticConventions.TOOL_DESCRIPTION] = description;
  }
  return attributes;
}

export function generateUniqueTraceId(
  eventType: string,
  traceId: string,
): string {
  return `${eventType}_${traceId.split("-").slice(0, 5).join("-")}`;
}

/**
 * Extracts provider attributes for LLM tracing.
 * Accepts either an object with a 'value' property or a string provider name.
 * Returns an attribute record with the provider name in lowercase.
 * @param provider - Provider object or string.
 * @returns Record of provider attributes.
 */
export function getLLMProviderAttributes(provider: unknown): Attributes {
  if (isObjectWithStringKeys(provider)) {
    const value = getStringAttributeValueFromUnknown(provider.value);
    return value ? { [SemanticConventions.LLM_PROVIDER]: value } : {};
  }
  if (typeof provider === "string") {
    return { [SemanticConventions.LLM_PROVIDER]: provider.toLowerCase() };
  }
  return {};
}

/**
 * Extracts message attributes for input/output messages, matching the Python _llm_messages_attributes logic.
 * Iterates over messages and extracts role, content, and content blocks if present.
 * @param messages - Array of message objects.
 * @param messageType - "input" or "output".
 * @returns Record of attribute keys and values.
 */
export function llmMessagesAttributes(
  messages: Message[] | undefined,
  messageType: "input" | "output",
): Attributes {
  const attributes: Attributes = {};
  const baseKey =
    messageType === "input"
      ? SemanticConventions.LLM_INPUT_MESSAGES
      : SemanticConventions.LLM_OUTPUT_MESSAGES;
  if (!Array.isArray(messages)) {
    return attributes;
  }
  for (let messageIndex = 0; messageIndex < messages.length; messageIndex++) {
    const message = messages[messageIndex];
    if (typeof message !== "object" || message === null) continue;
    if (message.role !== undefined) {
      attributes[
        `${baseKey}.${messageIndex}.${SemanticConventions.MESSAGE_ROLE}`
      ] = message.role;
    }
    if (message.content !== undefined) {
      attributes[
        `${baseKey}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENT}`
      ] = message.content;
    }
    if (Array.isArray(message.contents)) {
      for (
        let contentBlockIndex = 0;
        contentBlockIndex < message.contents.length;
        contentBlockIndex++
      ) {
        const contentBlock = message.contents[contentBlockIndex];
        if (typeof contentBlock !== "object" || contentBlock === null) continue;
        if (contentBlock.type !== undefined) {
          attributes[
            `${baseKey}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentBlockIndex}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`
          ] = contentBlock.type;
          if (typeof contentBlock.text === "string") {
            attributes[
              `${baseKey}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentBlockIndex}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`
            ] = contentBlock.text;
          }
        }
        if (
          typeof contentBlock.image === "object" &&
          contentBlock.image !== null &&
          typeof contentBlock.image.url === "string"
        ) {
          attributes[
            `${baseKey}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentBlockIndex}.${SemanticConventions.MESSAGE_CONTENT_IMAGE}.${SemanticConventions.IMAGE_URL}`
          ] = contentBlock.image.url;
        }
      }
    }
    if (typeof message.tool_call_id === "string") {
      attributes[
        `${baseKey}.${messageIndex}.${SemanticConventions.MESSAGE_TOOL_CALL_ID}`
      ] = message.tool_call_id;
    }
    if (Array.isArray(message.tool_calls)) {
      for (
        let toolCallIndex = 0;
        toolCallIndex < message.tool_calls.length;
        toolCallIndex++
      ) {
        const toolCall = message.tool_calls[toolCallIndex];
        if (typeof toolCall !== "object" || toolCall === null) continue;
        if (toolCall.id !== undefined) {
          attributes[
            `${baseKey}.${messageIndex}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_ID}`
          ] = toolCall.id;
        }
        if (
          toolCall.function !== undefined &&
          typeof toolCall.function === "object"
        ) {
          const func = toolCall.function;
          if (typeof func.name === "string") {
            attributes[
              `${baseKey}.${messageIndex}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
            ] = func.name;
          }
          if (typeof func.arguments === "string") {
            attributes[
              `${baseKey}.${messageIndex}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
            ] = func.arguments;
          } else if (
            typeof func.arguments === "object" &&
            func.arguments !== null
          ) {
            attributes[
              `${baseKey}.${messageIndex}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
            ] = JSON.stringify(func.arguments);
          }
        }
      }
    }
  }
  return attributes;
}

/**
 * Extracts document attributes for retrieval tracing.
 * Includes document ID, content, score, and metadata.
 * @param index - Index of the document in the retrieval list.
 * @param ref - Reference object containing document metadata and content.
 * @returns Record of document attributes for tracing.
 */
export function getDocumentAttributes(
  index: number,
  ref: DocumentReference,
): Attributes {
  const attributes: Attributes = {};
  const baseKey = `${SemanticConventions.RETRIEVAL_DOCUMENTS}.${index}`;
  const documentId = ref?.metadata?.["x-amz-bedrock-kb-chunk-id"] ?? "";
  if (documentId) {
    attributes[`${baseKey}.${SemanticConventions.DOCUMENT_ID}`] = documentId;
  }
  const documentContent = ref?.content?.text;
  if (documentContent) {
    attributes[`${baseKey}.${SemanticConventions.DOCUMENT_CONTENT}`] =
      documentContent;
  }
  attributes[`${baseKey}.${SemanticConventions.DOCUMENT_SCORE}`] =
    ref?.score ?? 0.0;
  attributes[`${baseKey}.${SemanticConventions.DOCUMENT_METADATA}`] =
    JSON.stringify({
      location: ref?.location ?? {},
      metadata: ref?.metadata ?? {},
      type: ref?.content?.type,
    });
  return attributes;
}
