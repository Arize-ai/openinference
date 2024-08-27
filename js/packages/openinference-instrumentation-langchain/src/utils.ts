import { Attributes, diag } from "@opentelemetry/api";
import {
  assertUnreachable,
  isNonEmptyArray,
  isNumber,
  isObject,
  isString,
} from "./typeUtils";
import { isAttributeValue } from "@opentelemetry/core";
import {
  MimeType,
  OpenInferenceSpanKind,
  RetrievalAttributePostfixes,
  SemanticAttributePrefixes,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { Run } from "@langchain/core/tracers/base";
import {
  LLMMessage,
  LLMMessageFunctionCall,
  LLMMessageToolCalls,
  LLMMessagesAttributes,
  LLMParameterAttributes,
  PromptTemplateAttributes,
  RetrievalDocument,
  TokenCountAttributes,
  ToolAttributes,
} from "./types";
import { withSafety } from "@arizeai/openinference-core";

export const RETRIEVAL_DOCUMENTS =
  `${SemanticAttributePrefixes.retrieval}.${RetrievalAttributePostfixes.documents}` as const;

/**
 * Handler for any unexpected errors that occur during processing.
 */
const onError = (message: string) => (error: unknown) => {
  diag.warn(
    `OpenInference-LangChain: error processing langchain run, falling back to null. ${message}. ${error}`,
  );
};

const safelyJSONStringify = withSafety({
  fn: JSON.stringify,
  onError: onError("Error stringifying JSON"),
});

/**
 * Flattens a nested object into a single level object with keys as dot-separated paths.
 * Specifies elements in arrays with their index as part of the path.
 * @param attributes - Nested attributes to flatten.
 * @param baseKey - Base key to prepend to all keys.
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
  return OpenInferenceSpanKind.CHAIN;
}

/**
 * Formats the input or output of a langchain run into OpenInference attributes for a span.
 * @param ioConfig - The input or output of a langchain run and the type of IO
 * @param ioConfig.io - The input or output of a langchain run
 * @param ioConfig.ioType - The type of IO
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
    [valueAttribute]: safelyJSONStringify(io),
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
 * @param messageKwargs - The langchain message kwargs to extract the content from
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
 * @param additionalKwargs - The langchain message additional kwargs to extract the tool calls from
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
 * @param messageData - The langchain message data to parse
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
 * @param input - The input of a langchain run.
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
 * Gets the first generation of the output of a langchain run.
 * @param output - The output of a langchain run.
 * @returns The first generation of the output or null.
 */
function getFirstOutputGeneration(output: Run["outputs"]) {
  if (!isObject(output)) {
    return null;
  }
  const maybeGenerations = output.generations;
  if (!isNonEmptyArray(maybeGenerations)) {
    return null;
  }
  // Only support the first 'set' of generations
  const firstGeneration = maybeGenerations[0];
  if (!isNonEmptyArray(firstGeneration)) {
    return null;
  }
  return firstGeneration;
}

/**
 * Formats the output messages of a langchain run into OpenInference attributes.
 * @param output - The output of a langchain run.
 * @returns The OpenInference attributes for the output messages.
 */
function formatOutputMessages(
  output: Run["outputs"],
): LLMMessagesAttributes | null {
  const firstGeneration = getFirstOutputGeneration(output);
  if (firstGeneration == null) {
    return null;
  }
  const parsedMessages: LLMMessage[] = [];
  firstGeneration.forEach((generation) => {
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

/**
 * Parses a langchain retrieval document into OpenInference attributes.
 * @param document - The langchain retrieval document to parse
 * @returns The OpenInference attributes for the retrieval document
 */
function parseRetrievalDocument(document: unknown) {
  if (!isObject(document)) {
    return null;
  }
  const parsedDocument: RetrievalDocument = {};
  if (isString(document.pageContent)) {
    parsedDocument["document.content"] = document.pageContent;
  }
  if (isObject(document.metadata)) {
    parsedDocument["document.metadata"] =
      safelyJSONStringify(document.metadata) ?? undefined;
  }
  return parsedDocument;
}

/**
 * Formats the retrieval documents of a langchain run into OpenInference attributes.
 * @param run - The langchain run to extract the retrieval documents from
 * @returns The OpenInference attributes for the retrieval documents.
 */
function formatRetrievalDocuments(run: Run) {
  const normalizedRunType = run.run_type.toLowerCase();
  if (normalizedRunType !== "retriever") {
    return null;
  }
  if (!isObject(run.outputs) || !Array.isArray(run.outputs.documents)) {
    return null;
  }
  return {
    [RETRIEVAL_DOCUMENTS]: run.outputs.documents
      .map(parseRetrievalDocument)
      .filter((doc) => doc != null),
  };
}

/**
 * Gets the model name from the langchain run extra data.
 * @param runExtra - The extra data from a langchain run
 * @returns The OpenInference attributes for the model name
 */
function formatLLMParams(
  runExtra: Run["extra"],
): LLMParameterAttributes | null {
  if (!isObject(runExtra) || !isObject(runExtra.invocation_params)) {
    return null;
  }
  const openInferenceParams: LLMParameterAttributes = {};

  openInferenceParams[SemanticConventions.LLM_INVOCATION_PARAMETERS] =
    safelyJSONStringify(runExtra.invocation_params) ?? undefined;

  if (isString(runExtra.invocation_params.model_name)) {
    openInferenceParams[SemanticConventions.LLM_MODEL_NAME] =
      runExtra.invocation_params.model_name;
  } else if (isString(runExtra.invocation_params.model)) {
    openInferenceParams[SemanticConventions.LLM_MODEL_NAME] =
      runExtra.invocation_params.model;
  }
  return openInferenceParams;
}

function getTemplateFromSerialized(serialized: Run["serialized"]) {
  if (!isObject(serialized) || !isObject(serialized.kwargs)) {
    return null;
  }
  const messages = serialized.kwargs.messages;
  if (!isNonEmptyArray(messages)) {
    return null;
  }
  const firstMessage = messages[0];
  if (!isObject(firstMessage) || !isObject(firstMessage.prompt)) {
    return null;
  }
  const template = firstMessage.prompt.template;
  if (!isString(template)) {
    return null;
  }
  return template;
}

const safelyGetTemplateFromSerialized = withSafety({
  fn: getTemplateFromSerialized,
});

/**
 * A best effort function to extract the prompt template from a langchain run.
 * @param run - The langchain run to extract the prompt template from
 * @returns The OpenInference attributes for the prompt template
 */
function formatPromptTemplate(run: Run): PromptTemplateAttributes | null {
  if (run.run_type.toLowerCase() !== "prompt") {
    return null;
  }
  return {
    [SemanticConventions.PROMPT_TEMPLATE_VARIABLES]:
      safelyJSONStringify(run.inputs) ?? undefined,
    [SemanticConventions.PROMPT_TEMPLATE_TEMPLATE]:
      safelyGetTemplateFromSerialized(run.serialized) ?? undefined,
  };
}

function getTokenCount(maybeCount: unknown) {
  return isNumber(maybeCount) ? maybeCount : undefined;
}

/**
 * Formats the token counts of a langchain run into OpenInference attributes.
 * @param outputs - The outputs of a langchain run
 * @returns The OpenInference attributes for the token counts
 */
function formatTokenCounts(
  outputs: Run["outputs"],
): TokenCountAttributes | null {
  if (!isObject(outputs)) {
    return null;
  }
  const llmOutput = outputs.llmOutput;
  if (!isObject(llmOutput)) {
    return null;
  }
  if (isObject(llmOutput.tokenUsage)) {
    return {
      [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: getTokenCount(
        llmOutput.tokenUsage.completionTokens,
      ),
      [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: getTokenCount(
        llmOutput.tokenUsage.promptTokens,
      ),
      [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]: getTokenCount(
        llmOutput.tokenUsage.totalTokens,
      ),
    };
  }
  /**
   * In the case of streamed outputs, the token counts are not available
   * only estimated counts provided by langchain (not the model provider) are available
   */
  if (isObject(llmOutput.estimatedTokenUsage)) {
    return {
      [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: getTokenCount(
        llmOutput.estimatedTokenUsage.completionTokens,
      ),
      [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: getTokenCount(
        llmOutput.estimatedTokenUsage.promptTokens,
      ),
      [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]: getTokenCount(
        llmOutput.estimatedTokenUsage.totalTokens,
      ),
    };
  }
  return null;
}

/**
 * Formats the function calls of a langchain run into OpenInference attributes.
 * @param outputs - The outputs of a langchain run
 * @returns The OpenInference attributes for the function calls
 */
function formatFunctionCalls(outputs: Run["outputs"]) {
  const firstGeneration = getFirstOutputGeneration(outputs);
  if (firstGeneration == null) {
    return null;
  }
  const maybeGeneration = firstGeneration[0];
  if (!isObject(maybeGeneration) || !isObject(maybeGeneration.message)) {
    return null;
  }

  const additionalKwargs = maybeGeneration.message.additional_kwargs;

  if (
    !isObject(additionalKwargs) ||
    !isObject(additionalKwargs.function_call)
  ) {
    return null;
  }

  return {
    [SemanticConventions.LLM_FUNCTION_CALL]: safelyJSONStringify(
      additionalKwargs.function_call,
    ),
  };
}

/**
 * Formats the tool calls of a langchain run into OpenInference attributes.
 * @param run - The langchain run to extract the tool calls from
 * @returns The OpenInference attributes for the tool calls
 */
function formatToolCalls(run: Run) {
  const normalizedRunType = run.run_type.toLowerCase();
  if (normalizedRunType !== "tool") {
    return null;
  }
  const toolAttributes: ToolAttributes = {
    [SemanticConventions.TOOL_NAME]: run.name,
  };
  if (!isObject(run.serialized)) {
    return toolAttributes;
  }
  if (isString(run.serialized.name)) {
    toolAttributes[SemanticConventions.TOOL_NAME] = run.serialized.name;
  }
  if (isString(run.serialized.description)) {
    toolAttributes[SemanticConventions.TOOL_DESCRIPTION] =
      run.serialized.description;
  }
  return toolAttributes;
}

/**
 * Formats the metadata of a langchain run into OpenInference attributes.
 * @param run - The langchain run to extract the metadata from
 * @returns The OpenInference attributes for the metadata
 */
function formatMetadata(run: Run) {
  if (!isObject(run.extra) || !isObject(run.extra.metadata)) {
    return null;
  }
  return {
    metadata: safelyJSONStringify(run.extra.metadata),
  };
}

export const safelyFlattenAttributes = withSafety({
  fn: flattenAttributes,
  onError: onError("Error flattening attributes"),
});
export const safelyFormatIO = withSafety({
  fn: formatIO,
  onError: onError("Error formatting IO"),
});
export const safelyFormatInputMessages = withSafety({
  fn: formatInputMessages,
  onError: onError("Error formatting input messages"),
});
export const safelyFormatOutputMessages = withSafety({
  fn: formatOutputMessages,
  onError: onError("Error formatting output messages"),
});
export const safelyGetOpenInferenceSpanKindFromRunType = withSafety({
  fn: getOpenInferenceSpanKindFromRunType,
  onError: onError("Error getting OpenInference span kind from run type"),
});
export const safelyFormatRetrievalDocuments = withSafety({
  fn: formatRetrievalDocuments,
  onError: onError("Error formatting retrieval documents"),
});
export const safelyFormatLLMParams = withSafety({
  fn: formatLLMParams,
  onError: onError("Error formatting LLM params"),
});
export const safelyFormatPromptTemplate = withSafety({
  fn: formatPromptTemplate,
  onError: onError("Error formatting prompt template"),
});
export const safelyFormatTokenCounts = withSafety({
  fn: formatTokenCounts,
  onError: onError("Error formatting token counts"),
});
export const safelyFormatFunctionCalls = withSafety({
  fn: formatFunctionCalls,
  onError: onError("Error formatting function calls"),
});
export const safelyFormatToolCalls = withSafety({
  fn: formatToolCalls,
  onError: onError("Error formatting tool calls"),
});
export const safelyFormatMetadata = withSafety({
  fn: formatMetadata,
  onError: onError("Error formatting metadata"),
});
