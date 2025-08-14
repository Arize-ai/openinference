import {
  getInputAttributes,
  getLLMAttributes,
  getOutputAttributes,
  getToolAttributes,
  getLLMInputMessageAttributes,
  getLLMOutputMessageAttributes,
  getDocumentAttributes,
} from "./attributeUtils";
import { LLMProvider } from "@arizeai/openinference-semantic-conventions";
import { TokenCount, Message, ToolCall, ToolCallFunction } from "./types";
import {
  fixLooseJsonString,
  getObjectDataFromUnknown,
  parseSanitizedJson,
} from "../utils/jsonUtils";
import { StringKeyedObject } from "../types";
import {
  isObjectWithStringKeys,
  safelyJSONStringify,
} from "@arizeai/openinference-core";
import {
  CHUNK_TYPES,
  ChunkType,
  TRACE_EVENT_TYPES,
  TraceEventType,
} from "./constants";
import { isArrayOfObjectWithStringKeys } from "../utils/typeUtils";
import { Attributes } from "@opentelemetry/api";
import { isAttributeValue } from "@opentelemetry/core";

/**
 * Return the first matching event type key discovered in {@link traceData}.
 * @returns {TraceEventType | null} The first matching event type key or null if not found.
 */
export function getEventType(
  traceData: StringKeyedObject,
): TraceEventType | undefined {
  for (const eventType of TRACE_EVENT_TYPES) {
    if (eventType in traceData) return eventType;
  }
}

/**
 * Extract the trace ID from a trace object.
 * @returns {string | undefined} The trace ID or undefined if not found.
 */
export function extractTraceId(
  traceData: StringKeyedObject,
): string | undefined {
  const eventType = getEventType(traceData);
  if (eventType == null) {
    return;
  }
  const eventData = getObjectDataFromUnknown({
    data: traceData,
    key: eventType,
  });
  if (eventData === undefined || eventData === null) {
    return;
  }
  for (const chunkType of CHUNK_TYPES) {
    const chunkData = getObjectDataFromUnknown({
      data: eventData,
      key: chunkType,
    });
    if (chunkData !== null && typeof chunkData["traceId"] === "string") {
      return chunkData["traceId"];
    }
  }
}

/**
 * Return the first matching chunk type discovered in {@link eventData}.
 * @returns {ChunkType | undefined} The first matching chunk type or undefined if not found.
 */
export function getChunkType(
  eventData: StringKeyedObject,
): ChunkType | undefined {
  for (const chunkType of CHUNK_TYPES) {
    if (chunkType in eventData) return chunkType;
  }
}

/**
 * Returns a string from an unknown value or null if it cannot be safely stringified.
 */
export function getStringAttributeValueFromUnknown(
  value: unknown,
): string | null {
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
              messageContent =
                getStringAttributeValueFromUnknown(maybeContent) ??
                messageContent;
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
    const inputValue =
      getObjectDataFromUnknown({ data: actionGroup, key: "text" }) || "";
    if (inputValue) {
      return getInputAttributes(inputValue);
    }
  }

  const codeInterpreter = getObjectDataFromUnknown({
    data: invocationInput,
    key: "codeInterpreterInvocationInput",
  });
  if (codeInterpreter) {
    const inputValue =
      getObjectDataFromUnknown({ data: codeInterpreter, key: "code" }) || "";
    if (inputValue) {
      return getInputAttributes(inputValue);
    }
  }

  const kbLookup = getObjectDataFromUnknown({
    data: invocationInput,
    key: "knowledgeBaseLookupInput",
  });
  if (kbLookup) {
    const inputValue =
      getObjectDataFromUnknown({ data: kbLookup, key: "text" }) || "";
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
export function getMetadataAttributes(
  traceMetadata: StringKeyedObject,
): StringKeyedObject | null {
  const metadata: StringKeyedObject = {};
  const clientRequestId = isAttributeValue(traceMetadata.clientRequestId)
    ? traceMetadata.clientRequestId
    : getStringAttributeValueFromUnknown(traceMetadata.clientRequestId);
  if (clientRequestId != null) {
    metadata["client_request_id"] = clientRequestId;
  }

  if (traceMetadata?.operationTotalTimeMs) {
    metadata["operation_total_time_ms"] = traceMetadata.operationTotalTimeMs;
  }
  if (traceMetadata?.totalTimeMs) {
    metadata["total_time_ms"] = traceMetadata.totalTimeMs;
  }
  if (traceMetadata?.startTime) {
    metadata["start_time"] = getTimeAttributeValue(traceMetadata.startTime);
  }
  if (traceMetadata?.endTime) {
    metadata["end_time"] = getTimeAttributeValue(traceMetadata.endTime);
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
    inputText = getStringAttributeValueFromUnknown(
      modelInvocationInput["text"],
    );
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
  llmAttributes["outputMessages"] = getOutputMessages(
    modelInvocationOutput || {},
  );
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
  const rawResponse = outputParams.rawResponse;
  if (isObjectWithStringKeys(rawResponse) && rawResponse.content != null) {
    const stringContent = getStringAttributeValueFromUnknown(
      rawResponse.content,
    );
    if (stringContent) {
      return stringContent;
    }
  }

  const parsedResponse = outputParams?.parsedResponse;

  if (isObjectWithStringKeys(parsedResponse)) {
    const maybeText = getStringAttributeValueFromUnknown(parsedResponse.text);
    if (maybeText) {
      return maybeText;
    }
    const maybeRationale = getStringAttributeValueFromUnknown(
      parsedResponse.rationale,
    );
    if (maybeRationale) {
      return maybeRationale;
    }
  }
  return undefined;
}

/**
 * Get output messages from model invocation output.
 * @param modelInvocationOutput Model invocation output object.
 * @returns Array of output messages.
 */
function getOutputMessages(
  modelInvocationOutput: StringKeyedObject,
): Message[] | null {
  const messages: Message[] = [];
  const rawResponse = getObjectDataFromUnknown({
    data: modelInvocationOutput,
    key: "rawResponse",
  });

  const outputContent = rawResponse?.content;
  if (outputContent == null) {
    return null;
  }
  let parsedContent: unknown | null = null;
  if (typeof outputContent === "string") {
    parsedContent = parseSanitizedJson(outputContent) ?? outputContent;
    if (!isObjectWithStringKeys(parsedContent)) {
      messages.push({ content: outputContent, role: "assistant" });
      return messages;
    }
  }

  if (!isObjectWithStringKeys(parsedContent)) {
    const stringifiedContent =
      getStringAttributeValueFromUnknown(outputContent);
    if (stringifiedContent) {
      messages.push({ content: stringifiedContent, role: "assistant" });
    }
    return messages;
  }
  try {
    const contents = parsedContent.content;
    if (contents == null) {
      return null;
    }
    if (!Array.isArray(contents)) {
      return null;
    }
    for (const content of contents) {
      if (isObjectWithStringKeys(content)) {
        const message = getAttributesFromOutputMessage({
          message: content,
          role: typeof content.role === "string" ? content.role : "assistant",
        });
        if (message) {
          messages.push(message);
        }
      }
    }
    return messages;
  } catch (e) {
    messages.push({
      content: safelyJSONStringify(outputContent) ?? undefined,
      role: "assistant",
    });
    return messages;
  }
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
  const outputData =
    getObjectDataFromUnknown({ data: collaboratorOutput, key: "output" }) || {};
  const outputType = outputData?.type || "TEXT";
  let outputValue: string | null = null;
  if (outputType === "TEXT") {
    outputValue = getStringAttributeValueFromUnknown(outputData?.text) ?? null;
  } else if (outputType === "RETURN_CONTROL") {
    if (outputData?.returnControlPayload !== undefined) {
      outputValue = safelyJSONStringify(outputData?.returnControlPayload);
    }
  }
  const messages: Message[] = [
    { role: "assistant", content: outputValue ?? "" },
  ];
  const metadata = {
    agent_collaborator_name:
      getStringAttributeValueFromUnknown(
        collaboratorOutput.agentCollaboratorName,
      ) ?? undefined,
    agent_collaborator_alias_arn:
      getStringAttributeValueFromUnknown(
        collaboratorOutput.agentCollaboratorAliasArn,
      ) ?? undefined,
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
export function getAttributesFromInvocationInput(
  invocationInput: StringKeyedObject,
): Attributes {
  const maybeActionGroupInvocationInput = getObjectDataFromUnknown({
    data: invocationInput,
    key: "actionGroupInvocationInput",
  });
  if (maybeActionGroupInvocationInput) {
    return getAttributesFromActionGroupInvocationInput(
      maybeActionGroupInvocationInput,
    );
  }

  const maybeCodeInterpreterInvocationInput = getObjectDataFromUnknown({
    data: invocationInput,
    key: "codeInterpreterInvocationInput",
  });
  if (maybeCodeInterpreterInvocationInput) {
    return getAttributesFromCodeInterpreterInput(
      maybeCodeInterpreterInvocationInput,
    );
  }
  const maybeKnowledgeBaseLookupInput = getObjectDataFromUnknown({
    data: invocationInput,
    key: "knowledgeBaseLookupInput",
  });
  if (maybeKnowledgeBaseLookupInput) {
    return getAttributesFromKnowledgeBaseLookupInput(
      maybeKnowledgeBaseLookupInput,
    );
  }
  const maybeAgentCollaboratorInvocationInput = getObjectDataFromUnknown({
    data: invocationInput,
    key: "agentCollaboratorInvocationInput",
  });
  if (maybeAgentCollaboratorInvocationInput) {
    return getAttributesFromAgentCollaboratorInvocationInput(
      maybeAgentCollaboratorInvocationInput,
    );
  }
  return {};
}

/**
 * Extract attributes from action group invocation input.
 * Extracts tool call, messages, tool attributes, and metadata for action group invocation.
 */
function getAttributesFromActionGroupInvocationInput(
  actionInput: StringKeyedObject,
): Attributes {
  const name =
    getStringAttributeValueFromUnknown(actionInput?.function) ?? undefined;
  const parameters =
    getStringAttributeValueFromUnknown(actionInput?.parameters) ?? "{}";
  const description =
    getStringAttributeValueFromUnknown(actionInput?.description) ?? undefined;

  // Build tool call function and tool call
  const toolCallFunction: ToolCallFunction = {
    name,
    arguments: parameters,
  };
  const toolCalls: ToolCall[] = [{ id: "default", function: toolCallFunction }];
  const messages: Message[] = [
    { tool_call_id: "default", role: "tool", tool_calls: toolCalls },
  ];
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
    llmInvocationParameters["action_group_name"] = isAttributeValue(
      actionInput.actionGroupName,
    )
      ? actionInput.actionGroupName
      : (safelyJSONStringify(actionInput.actionGroupName) ?? undefined);
  }
  if (actionInput.executionType) {
    llmInvocationParameters["execution_type"] = isAttributeValue(
      actionInput.executionType,
    )
      ? actionInput.executionType
      : (safelyJSONStringify(actionInput.executionType) ?? undefined);
  }
  if (actionInput.invocationId) {
    llmInvocationParameters["invocation_id"] = isAttributeValue(
      actionInput.invocationId,
    )
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
    metadata: safelyJSONStringify(llmInvocationParameters) ?? undefined,
  };
}

/**
 * Extract attributes from code interpreter invocation input.
 * Extracts tool call, messages, tool attributes, and metadata for code interpreter invocation.
 */
function getAttributesFromCodeInterpreterInput(
  codeInput: StringKeyedObject,
): Attributes {
  const toolCallFunction = {
    name: "code_interpreter",
    arguments: {
      code: codeInput?.code ?? "",
      files: codeInput?.files ?? "",
    },
  };
  const toolCalls = [{ id: "default", function: toolCallFunction }];
  const messages = [
    { tool_call_id: "default", role: "tool", tool_calls: toolCalls },
  ];
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
function getAttributesFromKnowledgeBaseLookupInput(
  kbData: StringKeyedObject,
): Attributes {
  const metadata = {
    invocation_type: "knowledge_base_lookup",
    knowledge_base_id:
      getStringAttributeValueFromUnknown(kbData?.knowledgeBaseId) ?? undefined,
  };
  return {
    ...getInputAttributes(kbData?.text ?? ""),
    metadata: safelyJSONStringify(metadata) ?? undefined,
  };
}

/**
 * Extract attributes from agent collaborator invocation input.
 * Extracts content, builds messages, and adds metadata for agent collaborator invocation.
 */
function getAttributesFromAgentCollaboratorInvocationInput(
  input: StringKeyedObject,
): Attributes {
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
      getStringAttributeValueFromUnknown(input.agentCollaboratorName) ??
      undefined,
    agent_collaborator_alias_arn:
      getStringAttributeValueFromUnknown(input.agentCollaboratorAliasArn) ??
      undefined,
    input_type: isAttributeValue(inputType) ? inputType : undefined,
  };
  return {
    ...getInputAttributes(content),
    ...getLLMInputMessageAttributes(messages),
    metadata: safelyJSONStringify(metadata) ?? undefined,
  };
}

/**
 * Extract attributes from observation event.
 * Processes the observation event to extract output attributes.
 * @param observation The observation event object.
 * @returns A dictionary of extracted output attributes.
 */
export function getAttributesFromObservation(
  observation: StringKeyedObject,
): Attributes {
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
    return getAttributesFromCodeInterpreterOutput(
      maybeCodeInterpreterInvocationOutput,
    );
  }

  const maybeKnowledgeBaseLookupOutput = getObjectDataFromUnknown({
    data: observation,
    key: "knowledgeBaseLookupOutput",
  });
  if (maybeKnowledgeBaseLookupOutput) {
    const retrievedReferences =
      maybeKnowledgeBaseLookupOutput?.retrievedReferences ?? [];
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
export function getFailureTraceAttributes(
  traceData: StringKeyedObject,
): Attributes {
  let failureMessage = "";
  if (traceData?.failureCode) {
    failureMessage += `Failure Code: ${traceData.failureCode}\n`;
  }
  if (traceData?.failureReason) {
    failureMessage += `Failure Reason: ${traceData.failureReason}`;
  }
  if (failureMessage) {
    return getOutputAttributes(failureMessage);
  }
  return {};
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
      const observationConst =
        getObjectDataFromUnknown({ data: observation, key: event }) || {};
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
    const toolCallId =
      getStringAttributeValueFromUnknown(message?.id) ?? undefined;
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
  return null;
}
