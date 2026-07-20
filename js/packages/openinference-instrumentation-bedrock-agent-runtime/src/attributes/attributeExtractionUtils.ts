import type { Attributes } from "@opentelemetry/api";
import { isAttributeValue } from "@opentelemetry/core";

import {
  isObjectWithStringKeys,
  safelyJSONParse,
  safelyJSONStringify,
} from "@arizeai/openinference-core";
import { LLMProvider, SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import type { StringKeyedObject } from "../types";
import {
  fixLooseJsonString,
  getObjectDataFromUnknown,
  parseSanitizedJson,
} from "../utils/jsonUtils";
import { isArrayOfObjectWithStringKeys } from "../utils/typeUtils";
import {
  getDocumentAttributes,
  getInputAttributes,
  getLLMAttributes,
  getLLMInputMessageAttributes,
  getLLMOutputMessageAttributes,
  getOutputAttributes,
  getToolAttributes,
} from "./attributeUtils";
import type { ChunkType, TraceEventType } from "./constants";
import { CHUNK_TYPES, PolicyFilterType, PolicyType, TRACE_EVENT_TYPES } from "./constants";
import type {
  Message,
  MessageContent,
  ReasoningMessageContent,
  TokenCount,
  ToolCall,
  ToolCallFunction,
} from "./types";

/**
 * Return the first matching event type key discovered in {@link traceData}.
 * @returns {TraceEventType | undefined} The first matching trace event type key or undefined if not found.
 */
export function getEventType(traceData: StringKeyedObject): TraceEventType | undefined {
  for (const eventType of TRACE_EVENT_TYPES) {
    if (eventType in traceData) return eventType;
  }
}

/**
 * Extract the trace ID from a trace object.
 * @returns {string | undefined} The trace ID or undefined if not found.
 */
export function extractTraceId(traceData: StringKeyedObject): string | undefined {
  const eventType = getEventType(traceData);
  if (eventType == null) {
    return;
  }
  const eventData = getObjectDataFromUnknown({
    data: traceData,
    key: eventType,
  });
  if (!eventData) {
    return;
  }
  if (eventData && "traceId" in eventData && typeof eventData.traceId === "string") {
    return eventData.traceId;
  }

  for (const chunkType of CHUNK_TYPES) {
    const chunkData = getObjectDataFromUnknown({
      data: eventData,
      key: chunkType,
    });
    if (chunkData && "traceId" in chunkData && typeof chunkData["traceId"] === "string") {
      return chunkData["traceId"];
    }
  }
}

/**
 * Return the first matching chunk type discovered in {@link eventData}.
 * @returns {ChunkType | undefined} The first matching chunk type or undefined if not found.
 */
export function getChunkType(eventData: StringKeyedObject): ChunkType | undefined {
  for (const chunkType of CHUNK_TYPES) {
    if (chunkType in eventData) return chunkType;
  }
}

/**
 * Returns a string from an unknown value or null if it cannot be safely stringified.
 */
export function getStringAttributeValueFromUnknown(value: unknown): string | null {
  if (typeof value === "string") {
    return value;
  }
  return safelyJSONStringify(value);
}

/**
 * Extracts messages an object from a text.
 * Parses input text into a list of Message objects.
 */
export function getInputMessagesObject(text: string): Message[] {
  try {
    const messages: Message[] = [];
    const inputMessages = parseSanitizedJson(text);

    if (!isObjectWithStringKeys(inputMessages)) {
      return text ? [{ content: text }] : [];
    }
    const system = getStringAttributeValueFromUnknown(inputMessages?.system);
    if (system) {
      messages.push({
        content: system,
        role: "system",
      });
    }
    const msgArr = isArrayOfObjectWithStringKeys(inputMessages.messages)
      ? inputMessages.messages
      : [];
    msgArr.forEach((message) => {
      const role = typeof message.role === "string" ? message.role : undefined;
      if (typeof message.content === "string") {
        const parsedContents = fixLooseJsonString(message.content);
        for (const parsedContent of parsedContents) {
          let messageContent: string = message.content;
          if (isObjectWithStringKeys(parsedContent)) {
            // If the content has a type, use it as a key to get the content
            const maybeType = parsedContent.type;
            if (typeof maybeType === "string") {
              const maybeContent = parsedContent[maybeType];
              // If we are unable to get the content, use the original content
              messageContent = getStringAttributeValueFromUnknown(maybeContent) ?? messageContent;
            }
          } else {
            messageContent = parsedContent;
          }
          messages.push({ content: messageContent, role });
        }
      }
    });
    return messages;
  } catch {
    return text ? [{ content: text }] : [];
  }
}

/**
 * Extracts parent input attributes from invocation input.
 * Extracts input attributes from various types of invocation inputs
 * (action group, code interpreter, knowledge base lookup, agent collaborator)
 * to be set on the parent span.
 */
export function getParentInputAttributesFromInvocationInput(
  invocationInput: StringKeyedObject,
): Attributes | undefined {
  if (!isObjectWithStringKeys(invocationInput)) return {};

  const actionGroup = getObjectDataFromUnknown({
    data: invocationInput,
    key: "actionGroupInvocationInput",
  });
  if (actionGroup) {
    const inputValue = getObjectDataFromUnknown({ data: actionGroup, key: "text" }) || "";
    if (inputValue) {
      return getInputAttributes(inputValue);
    }
  }

  const codeInterpreter = getObjectDataFromUnknown({
    data: invocationInput,
    key: "codeInterpreterInvocationInput",
  });
  if (codeInterpreter) {
    const inputValue = getObjectDataFromUnknown({ data: codeInterpreter, key: "code" }) || "";
    if (inputValue) {
      return getInputAttributes(inputValue);
    }
  }

  const kbLookup = getObjectDataFromUnknown({
    data: invocationInput,
    key: "knowledgeBaseLookupInput",
  });
  if (kbLookup) {
    const inputValue = getObjectDataFromUnknown({ data: kbLookup, key: "text" }) || "";
    if (inputValue) {
      return getInputAttributes(inputValue);
    }
  }

  const agentCollaborator = getObjectDataFromUnknown({
    data: invocationInput,
    key: "agentCollaboratorInvocationInput",
  });
  if (agentCollaborator) {
    const inputData = getObjectDataFromUnknown({
      data: agentCollaborator,
      key: "input",
    });
    if (inputData) {
      const inputType = inputData["type"];
      if (inputType === "TEXT") {
        const inputValue = inputData["text"];
        if (inputValue) {
          return getInputAttributes(inputValue);
        }
      } else if (inputType === "RETURN_CONTROL") {
        const returnControlResults = inputData["returnControlResults"];
        if (returnControlResults) {
          const inputValue = JSON.stringify(returnControlResults);
          return getInputAttributes(inputValue);
        }
      }
    }
  }
  return {};
}

function getTimeAttributeValue(value: unknown): number | undefined {
  if (value instanceof Date) {
    return value.getTime();
  }
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "string") {
    const maybeTime = new Date(value).getTime();
    if (!isNaN(maybeTime)) {
      return maybeTime;
    }
  }
  return undefined;
}

/**
 * Extracts metadata attributes from trace metadata.
 * Converts timestamps to nanoseconds and normalizes keys.
 * @param traceMetadata The trace metadata object.
 * @returns A record of extracted metadata attributes.
 */
export function getMetadataAttributes(traceMetadata: StringKeyedObject): StringKeyedObject | null {
  const metadata: StringKeyedObject = {};
  const clientRequestId = isAttributeValue(traceMetadata.clientRequestId)
    ? traceMetadata.clientRequestId
    : getStringAttributeValueFromUnknown(traceMetadata.clientRequestId);
  if (clientRequestId != null) {
    metadata["clientRequestId"] = clientRequestId;
  }

  if (traceMetadata?.operationTotalTimeMs) {
    metadata["operationTotalTimeMs"] = traceMetadata.operationTotalTimeMs;
  }
  if (traceMetadata?.totalTimeMs) {
    metadata["totalTimeMs"] = traceMetadata.totalTimeMs;
  }
  if (traceMetadata?.startTime) {
    metadata["startTime"] = getTimeAttributeValue(traceMetadata.startTime);
  }
  if (traceMetadata?.endTime) {
    metadata["endTime"] = getTimeAttributeValue(traceMetadata.endTime);
  }
  return metadata;
}

export function getStartAndEndTimeFromMetadata(traceData: StringKeyedObject): {
  startTime: number | undefined;
  endTime: number | undefined;
} {
  const maybeEndTimeAttribute = getTimeAttributeValue(traceData?.endTime);

  const maybeStartTimeAttribute = getTimeAttributeValue(traceData?.startTime);
  return {
    startTime: maybeStartTimeAttribute,
    endTime: maybeEndTimeAttribute,
  };
}

/**
 * Extract attributes from model invocation input.
 * Processes the model invocation input to extract relevant attributes such as model name,
 * invocation parameters, and input messages. Combines these with LLM-specific and span kind attributes.
 * @param modelInvocationInput The model invocation input dictionary.
 * @returns A dictionary of extracted attributes.
 */
export function getAttributesFromModelInvocationInput(
  modelInvocationInput: StringKeyedObject,
): Attributes {
  const llmAttributes: StringKeyedObject = {};
  let inputText: string | null = null;
  if (modelInvocationInput && "text" in modelInvocationInput) {
    inputText = getStringAttributeValueFromUnknown(modelInvocationInput["text"]);
  }
  const modelName = getModelName(modelInvocationInput || {}, {});
  if (modelName) {
    llmAttributes["modelName"] = modelName;
  }
  if (modelInvocationInput?.inferenceConfiguration) {
    llmAttributes["invocationParameters"] = safelyJSONStringify(
      modelInvocationInput?.inferenceConfiguration,
    );
  }
  llmAttributes["inputMessages"] =
    inputText != null ? getInputMessagesObject(inputText) : undefined;
  llmAttributes["provider"] = LLMProvider.AWS;

  return {
    ...getLLMAttributes({ ...llmAttributes }),
    ...getInputAttributes(inputText),
  };
}

/**
 * Extract attributes from model invocation output.
 * Processes the model invocation output to extract relevant attributes such as model name,
 * invocation parameters, output messages, and token counts. Combine these attributes with
 * LLM-specific attributes and output attributes.
 * @param modelInvocationOutput The model invocation output dictionary.
 * @returns A dictionary of extracted attributes.
 */
export function getAttributesFromModelInvocationOutput(
  modelInvocationOutput: StringKeyedObject,
): Attributes {
  const llmAttributes: StringKeyedObject = {};
  const modelName = getModelName({}, modelInvocationOutput || {});
  if (modelName) {
    llmAttributes["modelName"] = modelName;
  }

  if (modelInvocationOutput?.inferenceConfiguration) {
    llmAttributes["invocationParameters"] = JSON.stringify(
      modelInvocationOutput.inferenceConfiguration,
    );
  }
  llmAttributes["outputMessages"] = getOutputMessages(modelInvocationOutput || {});
  llmAttributes["tokenCount"] = getTokenCounts(modelInvocationOutput);
  let requestAttributes = {
    ...getLLMAttributes({ ...llmAttributes }),
  };
  const outputValue = getOutputValue(modelInvocationOutput || {});
  if (outputValue) {
    requestAttributes = {
      ...requestAttributes,
      ...getOutputAttributes(outputValue),
    };
  }
  return requestAttributes;
}

/**
 * Get the model name from input or output parameters.
 * @param inputParams Model invocation input parameters.
 * @param outputParams Model invocation output parameters.
 * @returns Model name if found, otherwise undefined.
 */
function getModelName(
  inputParams: StringKeyedObject,
  outputParams: StringKeyedObject,
): string | undefined {
  // Try to get model name from inputParams["foundationModel"]
  const foundationModel = inputParams?.foundationModel;
  if (typeof foundationModel === "string") {
    return foundationModel;
  }

  // Try to get model name from outputParams["rawResponse"]["content"]
  const rawResponse = getObjectDataFromUnknown({
    data: outputParams,
    key: "rawResponse",
  });
  if (rawResponse) {
    const content = getObjectDataFromUnknown({
      data: rawResponse,
      key: "content",
    });
    if (content) {
      if (typeof content.model === "string") {
        return content.model;
      }
    }
  }
}

/**
 * Get the output value from model invocation output.
 * @param outputParams Model invocation output parameters.
 * @returns Output value as a string, or undefined if not found.
 */
function getOutputValue(outputParams: StringKeyedObject): string | undefined {
  const valueFromRawResponse = getValueFromRawResponse(outputParams);
  if (valueFromRawResponse) {
    return valueFromRawResponse;
  }

  const valueFromParsedResponse = getValueFromParsedResponse(outputParams);
  if (valueFromParsedResponse) {
    return valueFromParsedResponse;
  }

  return undefined;
}

function getValueFromRawResponse(outputParams: StringKeyedObject): string | undefined {
  const rawResponse = outputParams.rawResponse;
  if (!isObjectWithStringKeys(rawResponse) || rawResponse.content == null) {
    return undefined;
  }

  const stringContent = getStringAttributeValueFromUnknown(rawResponse.content);
  if (!stringContent) {
    return undefined;
  }

  const content = safelyJSONParse(stringContent);
  if (!isObjectWithStringKeys(content)) {
    return stringContent;
  }

  const output = getObjectDataFromUnknown({ data: content, key: "output" });
  if (!output) {
    return stringContent;
  }

  const message = getObjectDataFromUnknown({ data: output, key: "message" });
  if (!message) {
    return stringContent;
  }

  if (Array.isArray(message.content)) {
    const textBlock = message.content.find(
      (b: unknown) => isObjectWithStringKeys(b) && typeof b.text === "string",
    );
    if (isObjectWithStringKeys(textBlock) && typeof textBlock.text === "string") {
      return textBlock.text;
    }
    return stringContent;
  }
  const messageContent = message.content;
  if (isObjectWithStringKeys(messageContent) && typeof messageContent.text === "string") {
    return messageContent.text;
  }
  return stringContent;
}

function getValueFromParsedResponse(outputParams: StringKeyedObject): string | undefined {
  const parsedResponse = outputParams?.parsedResponse;
  if (!isObjectWithStringKeys(parsedResponse)) {
    return undefined;
  }

  const text = getStringAttributeValueFromUnknown(parsedResponse.text);
  return text || undefined;
}

/**
 * Extracts a ReasoningMessageContent from a Converse-normalized reasoningContent object.
 * Handles both reasoningText (text+signature) and redactedContent shapes.
 * Fields may be null in Converse-normalized payloads; null is treated as absent.
 */
function extractReasoningContentFromConverseBlock(
  reasoningContent: StringKeyedObject,
): ReasoningMessageContent | null {
  const reasoningText = reasoningContent.reasoningText;
  if (isObjectWithStringKeys(reasoningText) && typeof reasoningText.text === "string") {
    const block: ReasoningMessageContent = { type: "reasoning", text: reasoningText.text };
    if (typeof reasoningText.signature === "string") {
      block.signature = reasoningText.signature;
    }
    return block;
  }
  const redactedContent = reasoningContent.redactedContent;
  if (redactedContent instanceof Uint8Array && redactedContent.length > 0) {
    return { type: "reasoning", data: Buffer.from(redactedContent).toString("base64") };
  }
  if (typeof redactedContent === "string" && redactedContent) {
    return { type: "reasoning", data: redactedContent };
  }
  return null;
}

/**
 * Get output messages from model invocation output.
 * Handles both Anthropic-native ({content: [...]}) and Converse-normalized
 * ({output: {message: {content: [...]}}}) rawResponse shapes.
 * Also captures a structured top-level reasoningContent field (GPT-OSS style),
 * deduplicating against any duplicate reasoning blocks inside rawResponse.
 * All content blocks are merged into a single output Message.
 * @param modelInvocationOutput Model invocation output object.
 * @returns Single-element array containing the merged output message, or null.
 */
function getOutputMessages(modelInvocationOutput: StringKeyedObject): Message[] | null {
  const contentBlocks: MessageContent[] = [];
  const toolCalls: ToolCall[] = [];
  let role = "assistant";
  const hasStructuredReasoning = false;

  // Capture structured top-level reasoningContent first (e.g. GPT-OSS inline-agent traces).
  // When present, any duplicate reasoning block inside rawResponse is skipped below.

  // const topLevelReasoning = modelInvocationOutput.reasoningContent;
  // if (isObjectWithStringKeys(topLevelReasoning)) {
  //   const block = extractReasoningContentFromConverseBlock(topLevelReasoning);
  //   if (block) {
  //     contentBlocks.push(block);
  //     hasStructuredReasoning = true;
  //   }
  // }

  const rawResponse = getObjectDataFromUnknown({
    data: modelInvocationOutput,
    key: "rawResponse",
  });
  const outputContent = rawResponse?.content;

  if (outputContent != null) {
    let parsedData: unknown = null;
    if (typeof outputContent === "string") {
      parsedData = parseSanitizedJson(outputContent);
    }

    if (!isObjectWithStringKeys(parsedData)) {
      // JSON parse failed — treat entire content as plain text.
      const str =
        typeof outputContent === "string"
          ? outputContent
          : (safelyJSONStringify(outputContent) ?? undefined);
      if (str) {
        contentBlocks.push({ type: "text", text: str });
      }
    } else {
      // Extract the content block list from either rawResponse shape:
      //   a. Anthropic-native:       { content: [...], role: "assistant" }
      //   b. Converse-normalized:    { output: { message: { role: "assistant", content: [...] } } }
      let rawContentBlocks: unknown[] = [];
      if (Array.isArray(parsedData.content)) {
        rawContentBlocks = parsedData.content;
        if (typeof parsedData.role === "string") role = parsedData.role;
      } else {
        const outputObj = getObjectDataFromUnknown({ data: parsedData, key: "output" });
        const messageObj = outputObj
          ? getObjectDataFromUnknown({ data: outputObj, key: "message" })
          : null;
        if (messageObj && Array.isArray(messageObj.content)) {
          rawContentBlocks = messageObj.content;
          if (typeof messageObj.role === "string") role = messageObj.role;
        }
      }

      for (const rawBlock of rawContentBlocks) {
        if (!isObjectWithStringKeys(rawBlock)) continue;
        // Detect reasoning blocks across both rawResponse shapes for dedup.
        const isReasoningBlock =
          rawBlock.type === "thinking" ||
          rawBlock.type === "redacted_thinking" ||
          (rawBlock.reasoningContent != null && isObjectWithStringKeys(rawBlock.reasoningContent));
        if (hasStructuredReasoning && isReasoningBlock) continue;

        const blockMessage = getAttributesFromOutputMessage({ message: rawBlock, role });
        if (!blockMessage) continue;
        if (Array.isArray(blockMessage.contents)) {
          contentBlocks.push(...(blockMessage.contents as MessageContent[]));
        } else if (typeof blockMessage.content === "string") {
          contentBlocks.push({ type: "text", text: blockMessage.content });
        }
        if (Array.isArray(blockMessage.tool_calls)) {
          toolCalls.push(...blockMessage.tool_calls);
        }
      }
    }
  }

  if (contentBlocks.length === 0 && toolCalls.length === 0) return null;

  const message: Message = { role };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  // Single plain text with no tool calls → use flat content for compatibility.
  if (contentBlocks.length === 1 && toolCalls.length === 0 && contentBlocks[0].type === "text") {
    message.content = (contentBlocks[0] as { type: "text"; text: string }).text;
  } else if (contentBlocks.length > 0) {
    message.contents = contentBlocks;
  }
  return [message];
}

/**
 * Extract token counts from model invocation output.
 * @param outputParams Model invocation output parameters.
 * @returns TokenCount object with prompt, completion, and total tokens.
 */
function getTokenCounts(outputParams: StringKeyedObject): TokenCount | null {
  const metadata = getObjectDataFromUnknown({
    data: outputParams,
    key: "metadata",
  });
  if (metadata == null || !("usage" in metadata)) {
    return null;
  }
  const usage = getObjectDataFromUnknown({ data: metadata, key: "usage" });
  let input: number | undefined;
  let output: number | undefined;
  let total: number | undefined;
  if (typeof usage?.inputTokens === "number") {
    input = usage.inputTokens;
  }
  if (typeof usage?.outputTokens === "number") {
    output = usage.outputTokens;
  }
  if (input != null && output != null) {
    total = input + output;
  }
  return { prompt: input, completion: output, total: total };
}

function getAttributesFromAgentCollaboratorInvocationOutput(
  collaboratorOutput: StringKeyedObject,
): Attributes {
  const outputData = getObjectDataFromUnknown({ data: collaboratorOutput, key: "output" }) || {};
  const outputType = outputData?.type || "TEXT";
  let outputValue: string | null = null;
  if (outputType === "TEXT") {
    outputValue = getStringAttributeValueFromUnknown(outputData?.text) ?? null;
  } else if (outputType === "RETURN_CONTROL") {
    if (outputData?.returnControlPayload !== undefined) {
      outputValue = safelyJSONStringify(outputData?.returnControlPayload);
    }
  }
  const messages: Message[] = [{ role: "assistant", content: outputValue ?? "" }];
  const metadata = {
    agent_collaborator_name:
      getStringAttributeValueFromUnknown(collaboratorOutput.agentCollaboratorName) ?? undefined,
    agent_collaborator_alias_arn:
      getStringAttributeValueFromUnknown(collaboratorOutput.agentCollaboratorAliasArn) ?? undefined,
    output_type: outputType,
  };
  return {
    ...getOutputAttributes(outputValue),
    ...getLLMOutputMessageAttributes(messages),
    metadata: safelyJSONStringify(metadata) ?? undefined,
  };
}

/**
 * Extract attributes from invocation input.
 * Checks for specific invocation input types and delegates to their respective extractors.
 */
export function getAttributesFromInvocationInput(invocationInput: StringKeyedObject): Attributes {
  const maybeActionGroupInvocationInput = getObjectDataFromUnknown({
    data: invocationInput,
    key: "actionGroupInvocationInput",
  });
  if (maybeActionGroupInvocationInput) {
    return getAttributesFromActionGroupInvocationInput(maybeActionGroupInvocationInput);
  }

  const maybeCodeInterpreterInvocationInput = getObjectDataFromUnknown({
    data: invocationInput,
    key: "codeInterpreterInvocationInput",
  });
  if (maybeCodeInterpreterInvocationInput) {
    return getAttributesFromCodeInterpreterInput(maybeCodeInterpreterInvocationInput);
  }
  const maybeKnowledgeBaseLookupInput = getObjectDataFromUnknown({
    data: invocationInput,
    key: "knowledgeBaseLookupInput",
  });
  if (maybeKnowledgeBaseLookupInput) {
    return getAttributesFromKnowledgeBaseLookupInput(maybeKnowledgeBaseLookupInput);
  }
  const maybeAgentCollaboratorInvocationInput = getObjectDataFromUnknown({
    data: invocationInput,
    key: "agentCollaboratorInvocationInput",
  });
  if (maybeAgentCollaboratorInvocationInput) {
    return getAttributesFromAgentCollaboratorInvocationInput(maybeAgentCollaboratorInvocationInput);
  }
  return {};
}

/**
 * Extract attributes from action group invocation input.
 * Extracts tool call, messages, tool attributes, and metadata for action group invocation.
 */
function getAttributesFromActionGroupInvocationInput(actionInput: StringKeyedObject): Attributes {
  const name = getStringAttributeValueFromUnknown(actionInput?.function) ?? undefined;
  const parameters = getStringAttributeValueFromUnknown(actionInput?.parameters) ?? "{}";
  const description = getStringAttributeValueFromUnknown(actionInput?.description) ?? undefined;

  // Build tool call function and tool call
  const toolCallFunction: ToolCallFunction = {
    name,
    arguments: parameters,
  };
  const toolCalls: ToolCall[] = [{ id: "default", function: toolCallFunction }];
  const messages: Message[] = [{ tool_call_id: "default", role: "tool", tool_calls: toolCalls }];
  // Prepare tool attributes
  const toolAttributes = getToolAttributes({
    name,
    description,
    parameters,
  });
  // Prepare metadata
  const llmInvocationParameters: Attributes = {
    invocation_type: "action_group_invocation",
  };
  if (actionInput.actionGroupName) {
    llmInvocationParameters["action_group_name"] = isAttributeValue(actionInput.actionGroupName)
      ? actionInput.actionGroupName
      : (safelyJSONStringify(actionInput.actionGroupName) ?? undefined);
  }
  if (actionInput.executionType) {
    llmInvocationParameters["execution_type"] = isAttributeValue(actionInput.executionType)
      ? actionInput.executionType
      : (safelyJSONStringify(actionInput.executionType) ?? undefined);
  }
  if (actionInput.invocationId) {
    llmInvocationParameters["invocation_id"] = isAttributeValue(actionInput.invocationId)
      ? actionInput.invocationId
      : (safelyJSONStringify(actionInput.invocationId) ?? undefined);
  }
  if (actionInput.verb) {
    llmInvocationParameters["verb"] = isAttributeValue(actionInput.verb)
      ? actionInput.verb
      : (safelyJSONStringify(actionInput.verb) ?? undefined);
  }
  if (actionInput.apiPath) {
    llmInvocationParameters["api_path"] = isAttributeValue(actionInput.apiPath)
      ? actionInput.apiPath
      : (safelyJSONStringify(actionInput.apiPath) ?? undefined);
  }
  return {
    ...getLLMInputMessageAttributes(messages),
    ...toolAttributes,
    [SemanticConventions.TOOL_PARAMETERS]:
      safelyJSONStringify(llmInvocationParameters) ?? undefined,
  };
}

/**
 * Extract attributes from code interpreter invocation input.
 * Extracts tool call, messages, tool attributes, and metadata for code interpreter invocation.
 */
function getAttributesFromCodeInterpreterInput(codeInput: StringKeyedObject): Attributes {
  const toolCallFunction = {
    name: "code_interpreter",
    arguments: {
      code: codeInput?.code ?? "",
      files: codeInput?.files ?? "",
    },
  };
  const toolCalls = [{ id: "default", function: toolCallFunction }];
  const messages = [{ tool_call_id: "default", role: "tool", tool_calls: toolCalls }];
  const name = "code_interpreter";
  const description = "Executes code and returns results";
  const parameters = JSON.stringify({
    code: { type: "string", description: "Code to execute" },
  });
  const metadata = {
    invocation_type: "code_execution",
    execution_context: codeInput?.context ?? {},
  };
  return {
    ...getInputAttributes(codeInput?.code ?? ""),
    ...getLLMInputMessageAttributes(messages),
    ...getToolAttributes({ name, description, parameters }),
    metadata: safelyJSONStringify(metadata) ?? undefined,
  };
}

/**
 * Extract attributes from knowledge base lookup input.
 * Extracts input attributes and metadata for knowledge base lookup invocation.
 */
function getAttributesFromKnowledgeBaseLookupInput(kbData: StringKeyedObject): Attributes {
  const metadata = {
    invocation_type: "knowledge_base_lookup",
    knowledge_base_id: getStringAttributeValueFromUnknown(kbData?.knowledgeBaseId) ?? undefined,
  };
  return {
    ...getInputAttributes(kbData?.text ?? ""),
    metadata: safelyJSONStringify(metadata) ?? undefined,
  };
}

/**
 * Extract span attributes from agent collaborator invocation input.
 * Extracts content, builds messages, and adds metadata for agent collaborator invocation.
 */
function getAttributesFromAgentCollaboratorInvocationInput(input: StringKeyedObject): Attributes {
  const inputData = getObjectDataFromUnknown({ data: input, key: "input" });
  const inputType = inputData?.type || "TEXT";
  let content = "";
  if (inputType === "TEXT") {
    content = getStringAttributeValueFromUnknown(inputData?.text) ?? "";
  } else if (inputType === "RETURN_CONTROL") {
    if (inputData?.returnControlResults !== undefined) {
      content = safelyJSONStringify(inputData.returnControlResults) ?? "";
    }
  }
  const messages = [{ content, role: "assistant" }];
  const metadata: Attributes = {
    invocation_type: "agent_collaborator_invocation",
    agent_collaborator_name:
      getStringAttributeValueFromUnknown(input.agentCollaboratorName) ?? undefined,
    agent_collaborator_alias_arn:
      getStringAttributeValueFromUnknown(input.agentCollaboratorAliasArn) ?? undefined,
    input_type: isAttributeValue(inputType) ? inputType : undefined,
  };
  return {
    ...getInputAttributes(content),
    ...getLLMInputMessageAttributes(messages),
    metadata: safelyJSONStringify(metadata) ?? undefined,
  };
}

/**
 * Extract span attributes from observation event.
 * Processes the observation event to extract output attributes.
 * @param observation The observation event object.
 * @returns A dictionary of extracted output attributes.
 */
export function getAttributesFromObservation(observation: StringKeyedObject): Attributes {
  if (!observation || typeof observation !== "object") return {};
  if ("actionGroupInvocationOutput" in observation) {
    const toolOutput =
      getObjectDataFromUnknown({
        data: observation,
        key: "actionGroupInvocationOutput",
      }) || {};
    return getOutputAttributes(toolOutput?.text ?? "");
  }
  const maybeCodeInterpreterInvocationOutput = getObjectDataFromUnknown({
    data: observation,
    key: "codeInterpreterInvocationOutput",
  });
  if (maybeCodeInterpreterInvocationOutput) {
    return getAttributesFromCodeInterpreterOutput(maybeCodeInterpreterInvocationOutput);
  }

  const maybeKnowledgeBaseLookupOutput = getObjectDataFromUnknown({
    data: observation,
    key: "knowledgeBaseLookupOutput",
  });
  if (maybeKnowledgeBaseLookupOutput) {
    const retrievedReferences = maybeKnowledgeBaseLookupOutput?.retrievedReferences ?? [];
    if (isArrayOfObjectWithStringKeys(retrievedReferences)) {
      return getAttributesFromKnowledgeBaseLookupOutput(retrievedReferences);
    }
  }
  const maybeAgentCollaboratorInvocationOutput = getObjectDataFromUnknown({
    data: observation,
    key: "agentCollaboratorInvocationOutput",
  });
  if (maybeAgentCollaboratorInvocationOutput) {
    return getAttributesFromAgentCollaboratorInvocationOutput(
      maybeAgentCollaboratorInvocationOutput,
    );
  }

  return {};
}

/**
 * Extract attributes from code interpreter output.
 * Builds output attributes and tool messages for code execution results.
 * @param codeInvocationOutput Code interpreter output object.
 * @returns Output attributes and tool messages for code execution.
 */
function getAttributesFromCodeInterpreterOutput(
  codeInvocationOutput: StringKeyedObject,
): Attributes {
  let outputValue: unknown = null;
  let files: unknown = null;
  if (codeInvocationOutput?.executionOutput) {
    outputValue = codeInvocationOutput?.executionOutput;
  } else if (codeInvocationOutput?.executionError) {
    outputValue = codeInvocationOutput.executionError;
  } else if (codeInvocationOutput?.executionTimeout) {
    outputValue = "Execution Timeout Error";
  } else if (codeInvocationOutput?.files) {
    files = codeInvocationOutput.files;
    outputValue = safelyJSONStringify(files);
  }

  let content: string | undefined;

  if (files != null) {
    content = safelyJSONStringify(files) ?? undefined;
  } else if (outputValue != null) {
    content = getStringAttributeValueFromUnknown(outputValue) ?? undefined;
  }
  const messages = [{ role: "tool", content }];
  return {
    ...getOutputAttributes(outputValue),
    ...getLLMOutputMessageAttributes(messages),
  };
}

/**
 * Extract attributes from knowledge base lookup output.
 * Builds document attributes for each retrieved reference.
 * @param retrievedReferences Array of retrieved reference objects.
 * @returns Combined document attributes for all references.
 */
function getAttributesFromKnowledgeBaseLookupOutput(
  retrievedReferences: Array<StringKeyedObject>,
): Attributes {
  return retrievedReferences.reduce((acc: Attributes, ref, i) => {
    return {
      ...acc,
      ...getDocumentAttributes(i, ref),
    };
  }, {} as Attributes);
}

/**
 * Extract attributes from failure trace data.
 * Builds a failure message from code and reason, and returns output attributes.
 * @param traceData Failure trace data object.
 * @returns Output attributes for the failure message.
 */
export function getFailureTraceAttributes(traceData: StringKeyedObject): Attributes {
  let failureMessage = "";
  if (traceData?.failureCode && typeof traceData.failureCode === "string") {
    failureMessage += `Failure Code: ${traceData.failureCode}\n`;
  }
  if (traceData?.failureReason && typeof traceData.failureReason === "string") {
    failureMessage += `Failure Reason: ${traceData.failureReason}`;
  }
  if (failureMessage) {
    return getOutputAttributes(failureMessage);
  }
  return {};
}

/**
 * Determine whether an agent invocation was blocked by any intervening guardrails
 * @param guardrails Array of guardrail objects to check
 * @returns True if any guardrail is blocked, false otherwise
 */
export function isBlockedGuardrail(guardrails: StringKeyedObject[]): boolean {
  const policyChecks = [
    {
      policyType: PolicyType.CONTENT,
      policyFilters: [PolicyFilterType.FILTERS],
    },
    {
      policyType: PolicyType.SENSITIVE_INFORMATION,
      policyFilters: [PolicyFilterType.PII_ENTITIES, PolicyFilterType.REGEXES],
    },
    {
      policyType: PolicyType.TOPIC,
      policyFilters: [PolicyFilterType.TOPICS],
    },
    {
      policyType: PolicyType.WORD,
      policyFilters: [PolicyFilterType.CUSTOM_WORDS, PolicyFilterType.MANAGED_WORD_LISTS],
    },
  ];

  for (const guardrail of guardrails) {
    const inputAssessments = Array.isArray(guardrail.inputAssessments)
      ? guardrail.inputAssessments
      : [];
    const outputAssessments = Array.isArray(guardrail.outputAssessments)
      ? guardrail.outputAssessments
      : [];
    const assessments = [...inputAssessments, ...outputAssessments];

    for (const assessment of assessments) {
      for (const { policyType, policyFilters } of policyChecks) {
        if (isAssessmentBlocked({ assessment, policyType, policyFilters })) {
          return true;
        }
      }
    }
  }
  return false;
}

/**
 * Parses through guardrail assessment to determine if the action is BLOCKED
 * @param assessment The assessment object to check
 * @param policyType The type of policy to check
 * @param policyFilters Array of filter types to check
 * @returns True if the assessment is blocked, false otherwise
 */
function isAssessmentBlocked({
  assessment,
  policyType,
  policyFilters,
}: {
  assessment: StringKeyedObject;
  policyType: string;
  policyFilters: string[];
}): boolean {
  const policy = getObjectDataFromUnknown({ data: assessment, key: policyType }) || {};

  // Collect all filters from the specified policy types
  const filters: StringKeyedObject[] = [];
  for (const filterType of policyFilters) {
    const filterArray = isArrayOfObjectWithStringKeys(policy[filterType]) ? policy[filterType] : [];
    filters.push(...filterArray);
  }

  return filters.some((filter) => filter?.action === "BLOCKED");
}

/**
 * Extract metadata attributes from an observation event.
 * @param observation The observation object containing event data.
 * @returns Extracted metadata attributes or an empty object.
 */
export function extractMetadataAttributesFromObservation(
  observation: StringKeyedObject,
): StringKeyedObject | null {
  const events = [
    "actionGroupInvocationOutput",
    "codeInterpreterInvocationOutput",
    "knowledgeBaseLookupOutput",
    "agentCollaboratorInvocationOutput",
  ];
  for (const event of events) {
    if (event in observation && observation[event]) {
      const observationConst = getObjectDataFromUnknown({ data: observation, key: event }) || {};
      const metadata =
        getObjectDataFromUnknown({
          data: observationConst,
          key: "metadata",
        }) || {};
      return getMetadataAttributes(metadata);
    }
  }
  return null;
}

/**
 * Extract attributes from a message dictionary.
 * @param message The message object to extract attributes from.
 * @param role The role associated with the message.
 * @returns Message object if attributes can be extracted, otherwise null.
 */
function getAttributesFromOutputMessage({
  message,
  role,
}: {
  message: StringKeyedObject;
  role: string;
}): Message | null {
  const text = getStringAttributeValueFromUnknown(message?.text);
  if (message.type === "text" && text != null) {
    return {
      content: text,
      role,
    };
  }
  if (message?.type === "tool_use") {
    const toolCallFunction: ToolCallFunction = {
      name: getStringAttributeValueFromUnknown(message?.name) ?? undefined,
      arguments: getStringAttributeValueFromUnknown(message?.input) ?? "{}",
    };
    const toolCallId = getStringAttributeValueFromUnknown(message?.id) ?? undefined;
    const toolCalls: ToolCall[] = [
      {
        id: toolCallId,
        function: toolCallFunction,
      },
    ];
    return {
      tool_call_id: toolCallId,
      role: "tool",
      tool_calls: toolCalls,
    };
  }
  if (message.type === "thinking") {
    const thinkingText = getStringAttributeValueFromUnknown(message?.thinking) ?? undefined;
    const signature = getStringAttributeValueFromUnknown(message?.signature) ?? undefined;
    if (thinkingText != null || signature != null) {
      return {
        role,
        contents: [
          {
            type: "reasoning",
            ...(thinkingText != null ? { text: thinkingText } : {}),
            ...(signature != null ? { signature } : {}),
          },
        ],
      };
    }
  }
  if (message.type === "redacted_thinking") {
    const data = getStringAttributeValueFromUnknown(message?.data) ?? undefined;
    if (data != null) {
      return {
        role,
        contents: [
          {
            type: "reasoning",
            data,
          },
        ],
      };
    }
  }
  // Converse-normalized block shape: no 'type' discriminator; fields may be null.
  // Handles { text, reasoningContent, toolUse } where each field may be null/absent.
  if (message.type == null) {
    const textVal = message.text;
    if (typeof textVal === "string" && textVal.length > 0) {
      return { content: textVal, role };
    }
    const converseReasoning = message.reasoningContent;
    if (isObjectWithStringKeys(converseReasoning)) {
      const block = extractReasoningContentFromConverseBlock(converseReasoning);
      if (block) return { role, contents: [block] };
    }
    const toolUseVal = message.toolUse;
    if (isObjectWithStringKeys(toolUseVal)) {
      const toolUseId = getStringAttributeValueFromUnknown(toolUseVal.toolUseId) ?? undefined;
      const toolName = getStringAttributeValueFromUnknown(toolUseVal.name) ?? undefined;
      if (toolUseId && toolName) {
        const toolCallFunction: ToolCallFunction = {
          name: toolName,
          arguments:
            toolUseVal.input != null && typeof toolUseVal.input === "object"
              ? (toolUseVal.input as Record<string, unknown>)
              : "{}",
        };
        return {
          tool_call_id: toolUseId,
          role: "tool",
          tool_calls: [{ id: toolUseId, function: toolCallFunction }],
        };
      }
    }
  }
  return null;
}
