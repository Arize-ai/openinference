import {
  getInputAttributes,
  getSpanKindAttributes,
  getLLMAttributes,
  getOutputAttributes,
  getToolAttributes,
  getLLMInputMessageAttributes,
  getLLMOutputMessageAttributes,
  getDocumentAttributes,
} from "./attribute-utils";
import {
  OpenInferenceSpanKind,
  LLMProvider,
} from "@arizeai/openinference-semantic-conventions";
import { TokenCount, Message } from "./types";
import { safeJsonParse, fixLooseJsonString } from "../utils/json-utils";

/**
 * @module AttributeExtractor
 * @description
 * Utility helpers for pulling key fields (event type, trace-ID, chunk type)
 * from raw Bedrock Agent Runtime trace payloads. Centralising this logic keeps
 * parsing rules consistent across the code-base.
 */
export class AttributeExtractor {
  /** Ordered a list of known event wrapper keys inside a trace object. */
  private static readonly EVENT_TYPES = [
    "preProcessingTrace",
    "orchestrationTrace",
    "postProcessingTrace",
    "failureTrace",
  ] as const;

  /** Ordered a list of known chunk wrapper keys inside an event object. */
  private static readonly CHUNK_TYPES = [
    "invocationInput",
    "modelInvocationInput",
    "modelInvocationOutput",
    "agentCollaboratorInvocationInput",
    "agentCollaboratorInvocationOutput",
    "rationale",
    "observation",
  ] as const;

  /**
   * Return the first matching event type key discovered in {@link traceData}.
   * Defaults to `'unknownTrace'` when not found.
   */
  static getEventType(traceData: Record<string, any>): string {
    for (const eventType of this.EVENT_TYPES) {
      if (eventType in traceData) return eventType;
    }
    return "unknownTrace";
  }

  /**
   * Extract the trace ID from a trace object. Falls back to
   * `'unknown-trace-id'` when missing.
   */
  static extractTraceId(traceData: Record<string, any>): string {
    const eventType = this.getEventType(traceData);

    for (const chunkType of this.CHUNK_TYPES) {
      if (chunkType in traceData[eventType])
        return traceData[eventType][chunkType]?.traceId ?? "unknown-trace-id";
    }
    return "unknown-trace-id";
  }

  /**
   * Return the first matching chunk type discovered in {@link eventData}.
   * Defaults to `'unknownChunk'` when not found.
   */
  static getChunkType(eventData: Record<string, any>): string {
    for (const chunkType of this.CHUNK_TYPES) {
      if (chunkType in eventData) return chunkType;
    }
    return "unknownChunk";
  }

  /**
   * Extracts messages an object from a text.
   * Parses input text into a list of Message objects.
   */
  public static getMessagesObject(text: string): Message[] {
    const messages: import("./types").Message[] = [];
    try {
      const inputMessages = safeJsonParse(text);
      if (inputMessages && typeof inputMessages === "object") {
        if (inputMessages.system) {
          messages.push({ content: inputMessages.system, role: "system" });
        }
        const msgArr = Array.isArray(inputMessages.messages)
          ? inputMessages.messages
          : [];
        for (const message of msgArr) {
          const role = message.role || "";
          if (message.content) {
            const parsedContents = fixLooseJsonString(message.content) || [
              message.content,
            ];
            for (const parsedContent of parsedContents) {
              let messageContent = message.content;
              if (typeof parsedContent === "object" && parsedContent !== null) {
                if (parsedContent.type && parsedContent[parsedContent.type]) {
                  messageContent = parsedContent[parsedContent.type];
                }
              } else if (typeof parsedContent === "string") {
                messageContent = parsedContent;
              }
              messages.push({ content: messageContent, role });
            }
          }
        }
      } else {
        messages.push({ content: text, role: "assistant" });
      }
    } catch {
      return [{ content: text, role: "assistant" }];
    }
    return messages;
  }

  /**
   * Extracts parent input attributes from invocation input.
   * Extracts input attributes from various types of invocation inputs
   * (action group, code interpreter, knowledge base lookup, agent collaborator)
   * to be set on the parent span.
   */
  public static getParentInputAttributesFromInvocationInput(
    invocationInput: Record<string, any>,
  ): Record<string, any> | undefined {
    if (!invocationInput || typeof invocationInput !== "object") return {};

    const actionGroup = invocationInput["actionGroupInvocationInput"] || {};
    if (actionGroup && typeof actionGroup === "object") {
      const inputValue = actionGroup["text"] || "";
      if (inputValue) {
        return getInputAttributes(inputValue);
      }
    }

    const codeInterpreter =
      invocationInput["codeInterpreterInvocationInput"] || {};
    if (codeInterpreter && typeof codeInterpreter === "object") {
      const inputValue = codeInterpreter["code"] || "";
      if (inputValue) {
        return getInputAttributes(inputValue);
      }
    }

    const kbLookup = invocationInput["knowledgeBaseLookupInput"] || {};
    if (kbLookup && typeof kbLookup === "object") {
      const inputValue = kbLookup["text"] || "";
      if (inputValue) {
        return getInputAttributes(inputValue);
      }
    }

    const agentCollaborator =
      invocationInput["agentCollaboratorInvocationInput"] || {};
    if (agentCollaborator && typeof agentCollaborator === "object") {
      const inputData = agentCollaborator["input"] || {};
      if (inputData && typeof inputData === "object") {
        const inputType = inputData["type"];
        if (inputType === "TEXT") {
          const inputValue = inputData["text"] || "";
          if (inputValue) {
            return getInputAttributes(inputValue);
          }
        } else if (inputType === "RETURN_CONTROL") {
          const returnControlResults = inputData["returnControlResults"];
          if (returnControlResults !== undefined) {
            const inputValue = JSON.stringify(returnControlResults);
            return getInputAttributes(inputValue);
          }
        }
      }
    }
    return {};
  }

  /**
   * Extracts metadata attributes from observation metadata.
   * Converts timestamps to nanoseconds and normalizes keys.
   * @param traceMetadata The observation metadata object.
   * @returns A record of extracted metadata attributes.
   */
  public static getObservationMetadataAttributes(
    traceMetadata: Record<string, any>,
  ): Record<string, any> {
    const metadata: Record<string, any> = {};
    if (!traceMetadata || typeof traceMetadata !== "object") return metadata;
    if (traceMetadata.clientRequestId) {
      metadata["client_request_id"] = traceMetadata.clientRequestId;
    }
    if (traceMetadata.endTime) {
      // If endTime is a Date object, convert to nanoseconds
      if (traceMetadata.endTime instanceof Date) {
        metadata["end_time"] = traceMetadata.endTime.getTime();
      } else if (typeof traceMetadata.endTime === "number") {
        // If already a timestamp in ms, convert to ns
        metadata["end_time"] = traceMetadata.endTime;
      } else if (typeof traceMetadata.endTime === "string") {
        const date = new Date(traceMetadata.endTime);
        if (!isNaN(date.getTime())) {
          metadata["end_time"] = date.getTime();
        }
      }
    }
    if (traceMetadata.startTime) {
      if (traceMetadata.startTime instanceof Date) {
        metadata["start_time"] = traceMetadata.startTime.getTime();
      } else if (typeof traceMetadata.startTime === "number") {
        metadata["start_time"] = traceMetadata.startTime;
      } else if (typeof traceMetadata.startTime === "string") {
        const date = new Date(traceMetadata.startTime);
        if (!isNaN(date.getTime())) {
          metadata["start_time"] = date.getTime();
        }
      }
    }
    if (traceMetadata.operationTotalTimeMs !== undefined) {
      metadata["operation_total_time_ms"] = traceMetadata.operationTotalTimeMs;
    }
    if (traceMetadata.totalTimeMs !== undefined) {
      metadata["total_time_ms"] = traceMetadata.totalTimeMs;
    }
    return metadata;
  }

  /**
   * Extracts metadata attributes from trace metadata.
   * Converts timestamps to nanoseconds and normalizes keys.
   * @param traceMetadata The trace metadata object.
   * @returns A record of extracted metadata attributes.
   */
  public static getMetadataAttributes(
    traceMetadata: Record<string, any>,
  ): Record<string, any> {
    const metadata: Record<string, any> = {};
    if (!traceMetadata || typeof traceMetadata !== "object") return metadata;
    if (traceMetadata.clientRequestId) {
      metadata["client_request_id"] = traceMetadata.clientRequestId;
    }
    if (traceMetadata.endTime) {
      // If endTime is a Date object, convert to nanoseconds
      if (traceMetadata.endTime instanceof Date) {
        metadata["end_time"] = traceMetadata.endTime.getTime();
      } else if (typeof traceMetadata.endTime === "number") {
        // If already a timestamp in ms, convert to ns
        metadata["end_time"] = traceMetadata.endTime;
      } else if (typeof traceMetadata.endTime === "string") {
        const date = new Date(traceMetadata.endTime);
        if (!isNaN(date.getTime())) {
          metadata["end_time"] = date.getTime();
        }
      }
    }
    if (traceMetadata.startTime) {
      if (traceMetadata.startTime instanceof Date) {
        metadata["start_time"] = traceMetadata.startTime.getTime();
      } else if (typeof traceMetadata.startTime === "number") {
        metadata["start_time"] = traceMetadata.startTime;
      } else if (typeof traceMetadata.startTime === "string") {
        const date = new Date(traceMetadata.startTime);
        if (!isNaN(date.getTime())) {
          metadata["start_time"] = date.getTime();
        }
      }
    }
    if (traceMetadata.operationTotalTimeMs !== undefined) {
      metadata["operation_total_time_ms"] = traceMetadata.operationTotalTimeMs;
    }
    if (traceMetadata.totalTimeMs !== undefined) {
      metadata["total_time_ms"] = traceMetadata.totalTimeMs;
    }
    return metadata;
  }

  /**
   * Extract attributes from model invocation input.
   * Processes the model invocation input to extract relevant attributes such as model name,
   * invocation parameters, and input messages. Combines these with LLM-specific and span kind attributes.
   * @param modelInvocationInput The model invocation input dictionary.
   * @returns A dictionary of extracted attributes.
   */
  public static getAttributesFromModelInvocationInput(
    modelInvocationInput: Record<string, any>,
  ): Record<string, any> {
    const llmAttributes: Record<string, any> = {};
    // Get input text
    let inputText = "";
    if (modelInvocationInput && "text" in modelInvocationInput) {
      inputText = modelInvocationInput["text"];
    }
    // Get model name and invocation parameters
    const modelName = AttributeExtractor.getModelName(
      modelInvocationInput || {},
      {},
    );
    if (modelName) {
      llmAttributes["modelName"] = modelName;
    }
    if (modelInvocationInput?.inferenceConfiguration) {
      llmAttributes["invocationParameters"] = JSON.stringify(
        modelInvocationInput?.inferenceConfiguration,
      );
    }
    // Get input and output messages
    llmAttributes["inputMessages"] =
      AttributeExtractor.getMessagesObject(inputText);
    llmAttributes.provider = LLMProvider.AWS;
    // Set attributes
    return {
      ...getLLMAttributes({ ...llmAttributes }),
      ...getSpanKindAttributes(OpenInferenceSpanKind.LLM),
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
  public static getAttributesFromModelInvocationOutput(
    modelInvocationOutput: Record<string, any>,
  ): Record<string, any> {
    const llmAttributes: Record<string, any> = {};
    // Get model name and invocation parameters
    const modelName = AttributeExtractor.getModelName(
      {},
      modelInvocationOutput || {},
    );
    if (modelName) {
      llmAttributes["modelName"] = modelName;
    }

    if (modelInvocationOutput?.inferenceConfiguration) {
      llmAttributes["invocationParameters"] = JSON.stringify(
        modelInvocationOutput.inferenceConfiguration,
      );
    }

    // Get output messages
    llmAttributes["outputMessages"] = AttributeExtractor.getOutputMessages(
      modelInvocationOutput || {},
    );
    llmAttributes["tokenCount"] = AttributeExtractor.getTokenCounts(
      modelInvocationOutput || {},
    );

    // Set attributes
    let requestAttributes = {
      ...getLLMAttributes({ ...llmAttributes }),
    };

    // Set output value
    const outputValue = AttributeExtractor.getOutputValue(
      modelInvocationOutput || {},
    );
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
  public static getModelName(
    inputParams: Record<string, any>,
    outputParams: Record<string, any>,
  ): string | undefined {
    // Try to get model name from inputParams["foundationModel"]
    if (
      inputParams &&
      typeof inputParams === "object" &&
      "foundationModel" in inputParams
    ) {
      return String(inputParams["foundationModel"]);
    }
    // Try to get model name from outputParams["rawResponse"]["content"]
    if (
      outputParams &&
      typeof outputParams === "object" &&
      "rawResponse" in outputParams
    ) {
      const rawResponse = outputParams["rawResponse"];
      if (
        rawResponse &&
        typeof rawResponse === "object" &&
        "content" in rawResponse
      ) {
        const outputText = rawResponse["content"];
        try {
          const data = JSON.parse(String(outputText));
          if (data && typeof data === "object" && "model" in data) {
            return String(data["model"]);
          }
        } catch (e) {
          // Optionally log error if logger is available
        }
      }
    }
    return undefined;
  }

  /**
   * Get the output value from model invocation output.
   * @param outputParams Model invocation output parameters.
   * @returns Output value as a string, or undefined if not found.
   */
  public static getOutputValue(
    outputParams: Record<string, any>,
  ): string | undefined {
    const rawResponse = outputParams?.rawResponse;
    if (
      rawResponse &&
      typeof rawResponse === "object" &&
      rawResponse.content !== undefined
    ) {
      return String(rawResponse.content);
    }
    const parsedResponse = outputParams?.parsedResponse || {};
    if (parsedResponse.text !== undefined) {
      return String(parsedResponse.text);
    }
    if (parsedResponse.rationale !== undefined) {
      return String(parsedResponse.rationale);
    }
    return undefined;
  }

  /**
   * Get output messages from model invocation output.
   * @param modelInvocationOutput Model invocation output object.
   * @returns Array of output messages.
   */
  public static getOutputMessages(
    modelInvocationOutput: Record<string, any>,
  ): any[] {
    const messages: any[] = [];
    const rawResponse = modelInvocationOutput?.rawResponse;
    if (rawResponse && typeof rawResponse === "object") {
      const outputText = rawResponse.content;
      if (outputText !== undefined) {
        try {
          const data = JSON.parse(String(outputText));
          const contents = data?.content || [];
          for (const content of contents) {
            if (typeof content === "object" && content !== null) {
              messages.push(
                AttributeExtractor.getAttributesFromMessage(
                  content,
                  content.role || "assistant",
                ),
              );
            }
          }
        } catch (e) {
          messages.push({ content: String(outputText), role: "assistant" });
        }
      }
    }
    return messages;
  }

  /**
   * Extract token counts from model invocation output.
   * @param outputParams Model invocation output parameters.
   * @returns TokenCount object with prompt, completion, and total tokens.
   */
  public static getTokenCounts(outputParams: Record<string, any>): TokenCount {
    const metadata = outputParams?.metadata;
    if (!metadata || !metadata.usage) return {};
    const usage = metadata.usage;
    let input = 0,
      output = 0,
      total = 0;
    if (typeof usage.inputTokens === "number") {
      input = usage.inputTokens;
    }
    if (typeof usage.outputTokens === "number") {
      output = usage.outputTokens;
    }
    if (
      typeof usage.inputTokens === "number" &&
      typeof usage.outputTokens === "number"
    ) {
      total = usage.inputTokens + usage.outputTokens;
    }
    return { prompt: input, completion: output, total: total };
  }

  public static getAttributesFromAgentCollaboratorInvocationOutput(
    collaboratorOutput: Record<string, any>,
  ): Record<string, any> {
    const outputData = collaboratorOutput?.output || {};
    const outputType = outputData?.type || "TEXT";

    // Extract content based on an output type
    let outputValue = "";
    if (outputType === "TEXT") {
      outputValue = outputData.text || "";
    } else if (outputType === "RETURN_CONTROL") {
      if (outputData.returnControlPayload !== undefined) {
        outputValue = JSON.stringify(outputData.returnControlPayload);
      }
    }

    // Create a message
    const messages = [{ role: "assistant", content: outputValue }];

    // Create metadata
    const metadata = {
      agent_collaborator_name: collaboratorOutput.agentCollaboratorName,
      agent_collaborator_alias_arn:
        collaboratorOutput.agentCollaboratorAliasArn,
      output_type: outputType,
    };

    return {
      ...getOutputAttributes(outputValue),
      ...getLLMOutputMessageAttributes(messages),
      metadata,
    };
  }

  /**
   * Extract attributes from invocation input.
   * Checks for specific invocation input types and delegates to their respective extractors.
   */
  public static getAttributesFromInvocationInput(
    invocationInput: Record<string, any>,
  ): Record<string, any> {
    if (!invocationInput || typeof invocationInput !== "object") return {};
    if ("actionGroupInvocationInput" in invocationInput) {
      return (
        AttributeExtractor.getAttributesFromActionGroupInvocationInput(
          invocationInput["actionGroupInvocationInput"],
        ) || {}
      );
    }
    if ("codeInterpreterInvocationInput" in invocationInput) {
      return (
        AttributeExtractor.getAttributesFromCodeInterpreterInput(
          invocationInput["codeInterpreterInvocationInput"],
        ) || {}
      );
    }
    if ("knowledgeBaseLookupInput" in invocationInput) {
      return (
        AttributeExtractor.getAttributesFromKnowledgeBaseLookupInput(
          invocationInput["knowledgeBaseLookupInput"],
        ) || {}
      );
    }
    if ("agentCollaboratorInvocationInput" in invocationInput) {
      return (
        AttributeExtractor.getAttributesFromAgentCollaboratorInvocationInput(
          invocationInput["agentCollaboratorInvocationInput"],
        ) || {}
      );
    }
    return {};
  }

  /**
   * Extract attributes from action group invocation input.
   * Extracts tool call, messages, tool attributes, and metadata for action group invocation.
   */
  public static getAttributesFromActionGroupInvocationInput(
    actionInput: Record<string, any>,
  ): Record<string, any> {
    const name = actionInput?.function || "";
    const parameters = actionInput?.parameters ?? {};
    const description = actionInput?.description || "";
    // Build tool call function and tool call
    const toolCallFunction = {
      name,
      arguments: parameters,
    };
    const toolCalls = [{ id: "default", function: toolCallFunction }];
    const messages = [
      { tool_call_id: "default", role: "tool", tool_calls: toolCalls },
    ];
    // Prepare tool attributes
    const toolAttributes = getToolAttributes({
      name,
      description,
      parameters: JSON.stringify(parameters),
    });
    // Prepare metadata
    const llmInvocationParameters: Record<string, any> = {
      invocation_type: "action_group_invocation",
      action_group_name: actionInput?.actionGroupName,
      execution_type: actionInput?.executionType,
    };
    if (actionInput?.invocationId) {
      llmInvocationParameters["invocation_id"] = actionInput.invocationId;
    }
    if (actionInput?.verb) {
      llmInvocationParameters["verb"] = actionInput.verb;
    }
    if (actionInput?.apiPath) {
      llmInvocationParameters["api_path"] = actionInput.apiPath;
    }
    return {
      ...getSpanKindAttributes(OpenInferenceSpanKind.TOOL),
      ...getLLMInputMessageAttributes(messages),
      ...toolAttributes,
      metadata: llmInvocationParameters,
    };
  }

  /**
   * Extract attributes from code interpreter invocation input.
   * Extracts tool call, messages, tool attributes, and metadata for code interpreter invocation.
   */
  public static getAttributesFromCodeInterpreterInput(
    codeInput: Record<string, any>,
  ): Record<string, any> {
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
      ...getSpanKindAttributes(OpenInferenceSpanKind.TOOL),
      ...getLLMInputMessageAttributes(messages),
      ...getToolAttributes({ name, description, parameters }),
      metadata,
    };
  }

  /**
   * Extract attributes from knowledge base lookup input.
   * Extracts input attributes and metadata for knowledge base lookup invocation.
   */
  public static getAttributesFromKnowledgeBaseLookupInput(
    kbData: Record<string, any>,
  ): Record<string, any> {
    const metadata = {
      invocation_type: "knowledge_base_lookup",
      knowledge_base_id: kbData?.knowledgeBaseId,
    };
    return {
      ...getInputAttributes(kbData?.text ?? ""),
      ...getSpanKindAttributes(OpenInferenceSpanKind.RETRIEVER),
      metadata,
    };
  }

  /**
   * Extract attributes from agent collaborator invocation input.
   * Extracts content, builds messages, and adds metadata for agent collaborator invocation.
   */
  public static getAttributesFromAgentCollaboratorInvocationInput(
    input: Record<string, any>,
  ): Record<string, any> {
    const inputData = input?.input || {};
    const inputType = inputData?.type || "TEXT";
    let content = "";
    if (inputType === "TEXT") {
      content = inputData.text || "";
    } else if (inputType === "RETURN_CONTROL") {
      if (inputData.returnControlResults !== undefined) {
        content = JSON.stringify(inputData.returnControlResults);
      }
    }
    const messages = [{ content, role: "assistant" }];
    const metadata = {
      invocation_type: "agent_collaborator_invocation",
      agent_collaborator_name: input.agentCollaboratorName,
      agent_collaborator_alias_arn: input.agentCollaboratorAliasArn,
      input_type: inputType,
    };
    return {
      ...getSpanKindAttributes(OpenInferenceSpanKind.AGENT),
      ...getInputAttributes(content),
      ...getLLMInputMessageAttributes(messages),
      metadata,
    };
  }

  /**
   * Extract attributes from observation event.
   * Processes the observation event to extract output attributes.
   * @param observation The observation event object.
   * @returns A dictionary of extracted output attributes.
   */
  public static getAttributesFromObservation(
    observation: Record<string, any>,
  ): Record<string, any> {
    if (!observation || typeof observation !== "object") return {};
    if ("actionGroupInvocationOutput" in observation) {
      const toolOutput = observation["actionGroupInvocationOutput"];
      return getOutputAttributes(toolOutput?.text ?? "");
    }
    if ("codeInterpreterInvocationOutput" in observation) {
      return AttributeExtractor.getAttributesFromCodeInterpreterOutput(
        observation["codeInterpreterInvocationOutput"],
      );
    }
    if ("knowledgeBaseLookupOutput" in observation) {
      return AttributeExtractor.getAttributesFromKnowledgeBaseLookupOutput(
        observation["knowledgeBaseLookupOutput"]?.retrievedReferences ?? [],
      );
    }
    if ("agentCollaboratorInvocationOutput" in observation) {
      return AttributeExtractor.getAttributesFromAgentCollaboratorInvocationOutput(
        observation["agentCollaboratorInvocationOutput"],
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
  public static getAttributesFromCodeInterpreterOutput(
    codeInvocationOutput: Record<string, any>,
  ): Record<string, any> {
    let outputValue: any = null;
    let files: any = null;

    if (codeInvocationOutput?.executionOutput) {
      outputValue = codeInvocationOutput.executionOutput;
    } else if (codeInvocationOutput?.executionError) {
      outputValue = codeInvocationOutput.executionError;
    } else if (codeInvocationOutput?.executionTimeout) {
      outputValue = "Execution Timeout Error";
    } else if (codeInvocationOutput?.files) {
      files = codeInvocationOutput.files;
      outputValue = JSON.stringify(files);
    }

    const content = files
      ? JSON.stringify(files)
      : outputValue !== null && outputValue !== undefined
        ? String(outputValue)
        : "";
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
  public static getAttributesFromKnowledgeBaseLookupOutput(
    retrievedReferences: any[],
  ): Record<string, any> {
    const attributes: Record<string, any> = {};
    if (!Array.isArray(retrievedReferences)) return attributes;
    for (let i = 0; i < retrievedReferences.length; i++) {
      const ref = retrievedReferences[i];
      Object.assign(attributes, getDocumentAttributes(i, ref));
    }
    return attributes;
  }

  /**
   * Extract attributes from failure trace data.
   * Builds a failure message from code and reason, and returns output attributes.
   * @param traceData Failure trace data object.
   * @returns Output attributes for the failure message.
   */
  public static getFailureTraceAttributes(
    traceData: Record<string, any>,
  ): Record<string, any> {
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
  static extractMetadataAttributesFromObservation(
    observation: Record<string, any>,
  ): object {
    const events = [
      "actionGroupInvocationOutput",
      "codeInterpreterInvocationOutput",
      "knowledgeBaseLookupOutput",
      "agentCollaboratorInvocationOutput",
    ];
    for (const event of events) {
      if (event in observation && observation[event]) {
        return AttributeExtractor.getMetadataAttributes(
          observation[event].metadata ?? {},
        );
      }
    }
    return {};
  }

  /**
   * Extract attributes from a message dictionary.
   * @param message The message object to extract attributes from.
   * @param role The role associated with the message.
   * @returns Message object if attributes can be extracted, otherwise null.
   */
  public static getAttributesFromMessage(
    message: Record<string, any>,
    role: string,
  ): import("./types").Message | null {
    if (!message || typeof message !== "object") return null;
    if (message.type === "text") {
      return { content: message.text || "", role };
    }
    if (message.type === "tool_use") {
      const toolCallFunction = {
        name: message.name || "",
        arguments: message.input || {},
      };
      const toolCalls = [{ id: message.id || "", function: toolCallFunction }];
      return {
        tool_call_id: message.id || "",
        role: "tool",
        tool_calls: toolCalls,
      };
    }
    return {};
  }
}
