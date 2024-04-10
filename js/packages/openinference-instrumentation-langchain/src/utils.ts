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
 * Flattens nested attributes into a single level object
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

function formatIO({
  io,
  type,
}: {
  io: Run["inputs"] | Run["outputs"];
  type: "input" | "output";
}) {
  let valueAttribute: string;
  let mimeTypeAttribute: string;
  switch (type) {
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
      assertUnreachable(type);
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

function getRoleFromMessageData(
  messageData: Record<string, unknown>,
): string | undefined {
  const messageIds = messageData.lc_id;
  if (!isNonEmptyArray(messageIds)) {
    return;
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
}

function getContentFromMessageData(
  messageKwargs: Record<string, unknown>,
): string | undefined {
  return isString(messageKwargs.content) ? messageKwargs.content : undefined;
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

function parseMessage(messageData: Record<string, unknown>): LLMMessage {
  const message: LLMMessage = {};

  message[SemanticConventions.MESSAGE_ROLE] =
    getRoleFromMessageData(messageData);

  const messageKwargs = messageData.lc_kwargs;
  if (!isObject(messageKwargs)) {
    return message;
  }
  message[SemanticConventions.MESSAGE_CONTENT] =
    getContentFromMessageData(messageKwargs);

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
