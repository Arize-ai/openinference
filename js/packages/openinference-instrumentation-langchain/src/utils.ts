import { Attributes, diag } from "@opentelemetry/api";
import {
  assertUnreachable,
  isNonEmptyArray,
  isObject,
  isString,
} from "./typeUtils";
import { isAttributeValue } from "@opentelemetry/core";
import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { Run } from "@langchain/core/tracers/base";
import {
  GenericFunction,
  LLMMessage,
  LLMMessageFunctionCall,
  LLMMessageToolCalls,
  LLMMessagesAttributes,
  SafeFunction,
} from "./types";

export function withSafety<T extends GenericFunction>(fn: T): SafeFunction<T> {
  return (...args) => {
    try {
      return fn(...args);
    } catch (error) {
      diag.error(`Failed to get attributes for span: ${error}`);
      return null;
    }
  };
}

/**
 * Flattens a nested object into a single level object with keys as dot-separated paths.
 * Specifies elements in arrays with their index as part of the path.
 * @param attributes Nested attributes to flatten.
 * @param baseKey Base key to prepend to all keys.
 * @returns Flattened attributes
 */
function flattenAttributes(
  attributes: Record<string, unknown>,
  baseKey: string = "",
): Attributes {
  const result: Attributes = {};
  for (const key in attributes) {
    const newKey = baseKey ? `${baseKey}.${key}` : key;
    const value = attributes[key];

    if (value == null) {
      continue;
    }

    if (isObject(value)) {
      Object.assign(result, flattenAttributes(value, newKey));
    } else if (Array.isArray(value)) {
      value.forEach((item, index) => {
        if (isObject(item)) {
          Object.assign(result, flattenAttributes(item, `${newKey}.${index}`));
        } else {
          result[`${newKey}.${index}`] = item;
        }
      });
    } else if (isAttributeValue(value)) {
      result[newKey] = value;
    }
  }
  return result;
}

/**
 * Gets the OpenInferenceSpanKind based on the langchain run type.
 * @param runType - The langchain run type
 * @returns The OpenInferenceSpanKind based on the langchain run type or "UNKNOWN".
 */
function getOpenInferenceSpanKindFromRunType(runType: string) {
  const normalizedRunType = runType.toUpperCase();
  if (normalizedRunType.includes("AGENT")) {
    return OpenInferenceSpanKind.AGENT;
  }

  if (normalizedRunType in OpenInferenceSpanKind) {
    return OpenInferenceSpanKind[
      normalizedRunType as keyof typeof OpenInferenceSpanKind
    ];
  }
  return "UNKNOWN";
}

/**
 * Formats the input or output of a langchain run into OpenInference attributes for a span.
 * @param ioConfig - The input or output of a langchain run and the type of IO
 * @returns The formatted input or output attributes for the span
 */
function formatIO({
  io,
  ioType,
}: {
  io: Run["inputs"] | Run["outputs"];
  ioType: "input" | "output";
}) {
  let valueAttribute: string;
  let mimeTypeAttribute: string;
  switch (ioType) {
    case "input": {
      valueAttribute = SemanticConventions.INPUT_VALUE;
      mimeTypeAttribute = SemanticConventions.INPUT_MIME_TYPE;
      break;
    }
    case "output": {
      valueAttribute = SemanticConventions.OUTPUT_VALUE;
      mimeTypeAttribute = SemanticConventions.OUTPUT_MIME_TYPE;
      break;
    }
    default:
      assertUnreachable(ioType);
  }
  if (io == null) {
    return {};
  }
  const values = Object.values(io);
  if (values.length === 1 && typeof values[0] === "string") {
    return {
      [valueAttribute]: values[0],
      [mimeTypeAttribute]: MimeType.TEXT,
    };
  }

  return {
    [valueAttribute]: JSON.stringify(io),
    [mimeTypeAttribute]: MimeType.JSON,
  };
}

/**
 * Gets the role of a message from the langchain message data.
 * @param messageData - The langchain message data to extract the role from
 * @returns The role of the message or null
 */
function getRoleFromMessageData(
  messageData: Record<string, unknown>,
): string | null {
  const messageIds = messageData.lc_id;
  if (!isNonEmptyArray(messageIds)) {
    return null;
  }
  const langchainMessageClass = messageIds[messageIds.length - 1];
  const normalizedLangchainMessageClass = isString(langchainMessageClass)
    ? langchainMessageClass.toLowerCase()
    : "";
  if (normalizedLangchainMessageClass.includes("human")) {
    return "user";
  }
  if (normalizedLangchainMessageClass.includes("ai")) {
    return "assistant";
  }
  if (normalizedLangchainMessageClass.includes("system")) {
    return "system";
  }
  if (normalizedLangchainMessageClass.includes("function")) {
    return "function";
  }
  if (
    normalizedLangchainMessageClass.includes("chat") &&
    isObject(messageData.kwargs) &&
    isString(messageData.kwargs.role)
  ) {
    return messageData.kwargs.role;
  }
  return null;
}

/**
 * Gets the content of a message from the langchain message kwargs.
 * @param messageKwargs The langchain message kwargs to extract the content from
 * @returns The content of the message or null
 */
function getContentFromMessageData(
  messageKwargs: Record<string, unknown>,
): string | null {
  return isString(messageKwargs.content) ? messageKwargs.content : null;
}

function getFunctionCallDataFromAdditionalKwargs(
  additionalKwargs: Record<string, unknown>,
): LLMMessageFunctionCall {
  const functionCall = additionalKwargs.function_call;
  if (!isObject(functionCall)) {
    return {};
  }
  const functionCallName = isString(functionCall.name)
    ? functionCall.name
    : undefined;
  const functionCallArgs = isString(functionCall.args)
    ? functionCall.args
    : undefined;
  return {
    [SemanticConventions.MESSAGE_FUNCTION_CALL_NAME]: functionCallName,
    [SemanticConventions.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON]:
      functionCallArgs,
  };
}

/**
 * Gets the tool calls from the langchain message additional kwargs and formats them into OpenInference attributes.
 * @param additionalKwargs The langchain message additional kwargs to extract the tool calls from
 * @returns the OpenInference attributes for the tool calls
 */
function getToolCallDataFromAdditionalKwargs(
  additionalKwargs: Record<string, unknown>,
): LLMMessageToolCalls {
  const toolCalls = additionalKwargs.tool_calls;
  if (!Array.isArray(toolCalls)) {
    return {};
  }
  const formattedToolCalls = toolCalls.map((toolCall) => {
    if (!isObject(toolCall) && !isObject(toolCall.function)) {
      return {};
    }
    const toolCallName = isString(toolCall.function.name)
      ? toolCall.function.name
      : undefined;
    const toolCallArgs = isString(toolCall.function.arguments)
      ? toolCall.function.arguments
      : undefined;
    return {
      [SemanticConventions.TOOL_CALL_FUNCTION_NAME]: toolCallName,
      [SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON]: toolCallArgs,
    };
  });
  return {
    [SemanticConventions.MESSAGE_TOOL_CALLS]: formattedToolCalls,
  };
}

/**
 * Parses a langchain message into OpenInference attributes.
 * @param messageData The langchain message data to parse
 * @returns The OpenInference attributes for the message
 */
function parseMessage(messageData: Record<string, unknown>): LLMMessage {
  const message: LLMMessage = {};

  const maybeRole = getRoleFromMessageData(messageData);
  if (maybeRole != null) {
    message[SemanticConventions.MESSAGE_ROLE] = maybeRole;
  }

  const messageKwargs = messageData.lc_kwargs;
  if (!isObject(messageKwargs)) {
    return message;
  }
  const maybeContent = getContentFromMessageData(messageKwargs);
  if (maybeContent != null) {
    message[SemanticConventions.MESSAGE_CONTENT] = maybeContent;
  }

  const additionalKwargs = messageKwargs.additional_kwargs;
  if (!isObject(additionalKwargs)) {
    return message;
  }
  return {
    ...message,
    ...getFunctionCallDataFromAdditionalKwargs(additionalKwargs),
    ...getToolCallDataFromAdditionalKwargs(additionalKwargs),
  };
}

/**
 * Formats the input messages of a langchain run into OpenInference attributes.
 * @param input The input of a langchain run.
 * @returns The OpenInference attributes for the input messages.
 */
function formatInputMessages(
  input: Run["inputs"],
): LLMMessagesAttributes | null {
  const maybeMessages = input.messages;
  if (!isNonEmptyArray(maybeMessages)) {
    return null;
  }

  // Only support the first 'set' of messages
  const firstMessages = maybeMessages[0];
  if (!isNonEmptyArray(firstMessages)) {
    return null;
  }

  const parsedMessages: LLMMessage[] = [];
  firstMessages.forEach((messageData) => {
    if (!isObject(messageData)) {
      return;
    }
    parsedMessages.push(parseMessage(messageData));
  });

  if (parsedMessages.length > 0) {
    return { [SemanticConventions.LLM_INPUT_MESSAGES]: parsedMessages };
  }

  return null;
}

/**
 * Formats the output messages of a langchain run into OpenInference attributes.
 * @param output The output of a langchain run.
 * @returns The OpenInference attributes for the output messages.
 */
function formatOutputMessages(
  output: Run["outputs"],
): LLMMessagesAttributes | null {
  if (output == null) {
    return null;
  }

  const maybeGenerations = output.generations;

  if (!isNonEmptyArray(maybeGenerations)) {
    return null;
  }
  // Only support the first 'set' of generations
  const firstGenerations = maybeGenerations[0];
  if (!isNonEmptyArray(firstGenerations)) {
    return null;
  }

  const parsedMessages: LLMMessage[] = [];
  firstGenerations.forEach((generation) => {
    if (!isObject(generation) || !isObject(generation.message)) {
      return;
    }
    parsedMessages.push(parseMessage(generation.message));
  });

  if (parsedMessages.length > 0) {
    return { [SemanticConventions.LLM_OUTPUT_MESSAGES]: parsedMessages };
  }

  return null;
}

export const safelyFlattenAttributes = withSafety(flattenAttributes);
export const safelyFormatIO = withSafety(formatIO);
export const safelyFormatInputMessages = withSafety(formatInputMessages);
export const safelyFormatOutputMessages = withSafety(formatOutputMessages);
export const safelyGetOpenInferenceSpanKindFromRunType = withSafety(
  getOpenInferenceSpanKindFromRunType,
);
