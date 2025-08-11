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
import { TokenCount, Message, ParsedInput } from "./types";
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
  static getEventType(traceData: Record<string, unknown>): string {
    for (const eventType of this.EVENT_TYPES) {
      if (eventType in traceData) return eventType;
    }
    return "unknownTrace";
  }

  /**
   * Extract the trace ID from a trace object. Falls back to
   * `'unknown-trace-id'` when missing.
   */
  static extractTraceId(traceData: Record<string, unknown>): string {
    const eventType = this.getEventType(traceData);

    for (const chunkType of this.CHUNK_TYPES) {
      if (
        eventType in traceData &&
        typeof traceData[eventType] === "object" &&
        traceData[eventType] !== null &&
        chunkType in (traceData[eventType] as Record<string, unknown>)
      ) {
        const chunkObj = (traceData[eventType] as Record<string, unknown>)[
          chunkType
        ];
        if (chunkObj && typeof chunkObj === "object" && "traceId" in chunkObj) {
          return (
            ((chunkObj as Record<string, unknown>)["traceId"] as string) ??
            "unknown-trace-id"
          );
        }
      }
    }
    return "unknown-trace-id";
  }

  /**
   * Return the first matching chunk type discovered in {@link eventData}.
   * Defaults to `'unknownChunk'` when not found.
   */
  static getChunkType(eventData: Record<string, unknown>): string {
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
    const messages: Message[] = [];
    try {
      const inputMessages: ParsedInput = safeJsonParse(text);
      if (inputMessages) {
        if (inputMessages?.system) {
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
              let messageContent: string = message.content;
              if (typeof parsedContent === "object" && parsedContent !== null) {
                if (parsedContent.type && parsedContent[parsedContent.type]) {
                  messageContent = parsedContent[parsedContent.type] as string;
                }
              } else {
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
    invocationInput: Record<string, unknown>,
  ): Record<string, unknown> | undefined {
    if (!invocationInput || typeof invocationInput !== "object") return {};

    const actionGroup =
      (invocationInput as Record<string, unknown>)[
        "actionGroupInvocationInput"
      ] || {};
    if (actionGroup && typeof actionGroup === "object") {
      const inputValue = (actionGroup as Record<string, unknown>)["text"] || "";
      if (inputValue) {
        return getInputAttributes(inputValue as string);
      }
    }

    const codeInterpreter =
      (invocationInput as Record<string, unknown>)[
        "codeInterpreterInvocationInput"
      ] || {};
    if (codeInterpreter && typeof codeInterpreter === "object") {
      const inputValue =
        (codeInterpreter as Record<string, unknown>)["code"] || "";
      if (inputValue) {
        return getInputAttributes(inputValue as string);
      }
    }

    const kbLookup =
      (invocationInput as Record<string, unknown>)[
        "knowledgeBaseLookupInput"
      ] || {};
    if (kbLookup && typeof kbLookup === "object") {
      const inputValue = (kbLookup as Record<string, unknown>)["text"] || "";
      if (inputValue) {
        return getInputAttributes(inputValue as string);
      }
    }

    const agentCollaborator =
      (invocationInput as Record<string, unknown>)[
        "agentCollaboratorInvocationInput"
      ] || {};
    if (agentCollaborator && typeof agentCollaborator === "object") {
      const inputData =
        (agentCollaborator as Record<string, unknown>)["input"] || {};
      if (inputData && typeof inputData === "object") {
        const inputType = (inputData as Record<string, unknown>)["type"];
        if (inputType === "TEXT") {
          const inputValue =
            (inputData as Record<string, unknown>)["text"] || "";
          if (inputValue) {
            return getInputAttributes(inputValue as string);
          }
        } else if (inputType === "RETURN_CONTROL") {
          const returnControlResults = (inputData as Record<string, unknown>)[
            "returnControlResults"
          ];
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
    traceMetadata: Record<string, unknown>,
  ): Record<string, unknown> {
    const metadata: Record<string, unknown> = {};
    if (!traceMetadata || typeof traceMetadata !== "object") return metadata;
    if ((traceMetadata as Record<string, unknown>).clientRequestId) {
      metadata["client_request_id"] = (
        traceMetadata as Record<string, unknown>
      ).clientRequestId;
    }
    if ((traceMetadata as Record<string, unknown>).endTime) {
      const endTime = (traceMetadata as Record<string, unknown>).endTime;
      if (endTime instanceof Date) {
        metadata["end_time"] = endTime.getTime();
      } else if (typeof endTime === "number") {
        metadata["end_time"] = endTime;
      } else if (typeof endTime === "string") {
        const date = new Date(endTime);
        if (!isNaN(date.getTime())) {
          metadata["end_time"] = date.getTime();
        }
      }
    }
    if ((traceMetadata as Record<string, unknown>).startTime) {
      const startTime = (traceMetadata as Record<string, unknown>).startTime;
      if (startTime instanceof Date) {
        metadata["start_time"] = startTime.getTime();
      } else if (typeof startTime === "number") {
        metadata["start_time"] = startTime;
      } else if (typeof startTime === "string") {
        const date = new Date(startTime);
        if (!isNaN(date.getTime())) {
          metadata["start_time"] = date.getTime();
        }
      }
    }
    if (
      (traceMetadata as Record<string, unknown>).operationTotalTimeMs !==
      undefined
    ) {
      metadata["operation_total_time_ms"] = (
        traceMetadata as Record<string, unknown>
      ).operationTotalTimeMs;
    }
    if ((traceMetadata as Record<string, unknown>).totalTimeMs !== undefined) {
      metadata["total_time_ms"] = (
        traceMetadata as Record<string, unknown>
      ).totalTimeMs;
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
    traceMetadata: Record<string, unknown>,
  ): Record<string, unknown> {
    const metadata: Record<string, unknown> = {};
    if (!traceMetadata || typeof traceMetadata !== "object") return metadata;
    if ((traceMetadata as Record<string, unknown>).clientRequestId) {
      metadata["client_request_id"] = (
        traceMetadata as Record<string, unknown>
      ).clientRequestId;
    }
    if ((traceMetadata as Record<string, unknown>).endTime) {
      const endTime = (traceMetadata as Record<string, unknown>).endTime;
      if (endTime instanceof Date) {
        metadata["end_time"] = endTime.getTime();
      } else if (typeof endTime === "number") {
        metadata["end_time"] = endTime;
      } else if (typeof endTime === "string") {
        const date = new Date(endTime);
        if (!isNaN(date.getTime())) {
          metadata["end_time"] = date.getTime();
        }
      }
    }
    if ((traceMetadata as Record<string, unknown>).startTime) {
      const startTime = (traceMetadata as Record<string, unknown>).startTime;
      if (startTime instanceof Date) {
        metadata["start_time"] = startTime.getTime();
      } else if (typeof startTime === "number") {
        metadata["start_time"] = startTime;
      } else if (typeof startTime === "string") {
        const date = new Date(startTime);
        if (!isNaN(date.getTime())) {
          metadata["start_time"] = date.getTime();
        }
      }
    }
    if (
      (traceMetadata as Record<string, unknown>).operationTotalTimeMs !==
      undefined
    ) {
      metadata["operation_total_time_ms"] = (
        traceMetadata as Record<string, unknown>
      ).operationTotalTimeMs;
    }
    if ((traceMetadata as Record<string, unknown>).totalTimeMs !== undefined) {
      metadata["total_time_ms"] = (
        traceMetadata as Record<string, unknown>
      ).totalTimeMs;
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
    modelInvocationInput: Record<string, unknown>,
  ): Record<string, unknown> {
    const llmAttributes: Record<string, unknown> = {};
    let inputText = "";
    if (modelInvocationInput && "text" in modelInvocationInput) {
      inputText = modelInvocationInput["text"] as string;
    }
    const modelName = AttributeExtractor.getModelName(
      modelInvocationInput || {},
      {},
    );
    if (modelName) {
      llmAttributes["modelName"] = modelName;
    }
    if (
      (modelInvocationInput as Record<string, unknown>)?.inferenceConfiguration
    ) {
      llmAttributes["invocationParameters"] = JSON.stringify(
        (modelInvocationInput as Record<string, unknown>)
          ?.inferenceConfiguration,
      );
    }
    llmAttributes["inputMessages"] =
      AttributeExtractor.getMessagesObject(inputText);
    llmAttributes["provider"] = LLMProvider.AWS;
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
    modelInvocationOutput: Record<string, unknown>,
  ): Record<string, unknown> {
    const llmAttributes: Record<string, unknown> = {};
    const modelName = AttributeExtractor.getModelName(
      {},
      modelInvocationOutput || {},
    );
    if (modelName) {
      llmAttributes["modelName"] = modelName;
    }
    if (
      (modelInvocationOutput as Record<string, unknown>)?.inferenceConfiguration
    ) {
      llmAttributes["invocationParameters"] = JSON.stringify(
        (modelInvocationOutput as Record<string, unknown>)
          .inferenceConfiguration,
      );
    }
    llmAttributes["outputMessages"] = AttributeExtractor.getOutputMessages(
      modelInvocationOutput || {},
    );
    llmAttributes["tokenCount"] = AttributeExtractor.getTokenCounts(
      modelInvocationOutput || {},
    );
    let requestAttributes = {
      ...getLLMAttributes({ ...llmAttributes }),
    };
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
    inputParams: Record<string, unknown>,
    outputParams: Record<string, unknown>,
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
    outputParams: Record<string, unknown>,
  ): string | undefined {
    const rawResponse = outputParams?.rawResponse || {};
    if (typeof rawResponse === "object" && rawResponse !== null) {
      if ("content" in rawResponse && rawResponse.content !== undefined) {
        return String(rawResponse.content);
      }
    }
    const parsedResponse = outputParams?.parsedResponse || {};
    if (typeof parsedResponse === "object" && parsedResponse !== null) {
      if ("text" in parsedResponse && parsedResponse.text !== undefined) {
        return String(parsedResponse.text);
      }
      if (
        "rationale" in parsedResponse &&
        parsedResponse.rationale !== undefined
      ) {
        return String(parsedResponse.rationale);
      }
    }
    return undefined;
  }

  /**
   * Get output messages from model invocation output.
   * @param modelInvocationOutput Model invocation output object.
   * @returns Array of output messages.
   */
  public static getOutputMessages(
    modelInvocationOutput: Record<string, unknown>,
  ): Record<string, unknown>[] {
    const messages: Record<string, unknown>[] = [];
    const rawResponse = modelInvocationOutput?.rawResponse;
    if (rawResponse && typeof rawResponse === "object") {
      const rawObj = rawResponse as Record<string, unknown>;
      if ("content" in rawObj && rawObj.content !== undefined) {
        const outputText = rawObj.content;
        try {
          const data = JSON.parse(String(outputText));
          const contents = data?.content || [];
          for (const content of contents) {
            if (typeof content === "object" && content !== null) {
              messages.push(
                AttributeExtractor.getAttributesFromMessage(
                  content,
                  ((content as Record<string, unknown>).role as string) ||
                    "assistant",
                ) as Record<string, unknown>,
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
  public static getTokenCounts(
    outputParams: Record<string, unknown>,
  ): TokenCount {
    const metadata = outputParams?.metadata;
    if (!metadata || typeof metadata !== "object" || !("usage" in metadata))
      return {};
    const usage = (metadata as Record<string, unknown>).usage as Record<
      string,
      unknown
    >;
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
    collaboratorOutput: Record<string, unknown>,
  ): Record<string, unknown> {
    const outputData = collaboratorOutput?.output || {};
    const outputType = (outputData as Record<string, unknown>)?.type || "TEXT";
    let outputValue: unknown = "";
    if (outputType === "TEXT") {
      outputValue = (outputData as Record<string, unknown>).text || "";
    } else if (outputType === "RETURN_CONTROL") {
      if (
        (outputData as Record<string, unknown>).returnControlPayload !==
        undefined
      ) {
        outputValue = JSON.stringify(
          (outputData as Record<string, unknown>).returnControlPayload,
        );
      }
    }
    const messages: Message[] = [
      { role: "assistant", content: outputValue as string },
    ];
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
    invocationInput: Record<string, unknown>,
  ): Record<string, unknown> {
    if (!invocationInput || typeof invocationInput !== "object") return {};
    if ("actionGroupInvocationInput" in invocationInput) {
      return (
        AttributeExtractor.getAttributesFromActionGroupInvocationInput(
          invocationInput["actionGroupInvocationInput"] as Record<
            string,
            string
          >,
        ) || {}
      );
    }
    if ("codeInterpreterInvocationInput" in invocationInput) {
      return (
        AttributeExtractor.getAttributesFromCodeInterpreterInput(
          invocationInput["codeInterpreterInvocationInput"] as Record<
            string,
            unknown
          >,
        ) || {}
      );
    }
    if ("knowledgeBaseLookupInput" in invocationInput) {
      return (
        AttributeExtractor.getAttributesFromKnowledgeBaseLookupInput(
          invocationInput["knowledgeBaseLookupInput"] as Record<
            string,
            unknown
          >,
        ) || {}
      );
    }
    if ("agentCollaboratorInvocationInput" in invocationInput) {
      return (
        AttributeExtractor.getAttributesFromAgentCollaboratorInvocationInput(
          invocationInput["agentCollaboratorInvocationInput"] as Record<
            string,
            unknown
          >,
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
    actionInput: Record<string, string>,
  ): Record<string, unknown> {
    const name: string = actionInput?.function;
    const parameters = actionInput?.parameters ?? {};
    const description: string = actionInput?.description;
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
    const llmInvocationParameters: Record<string, unknown> = {
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
    codeInput: Record<string, unknown>,
  ): Record<string, unknown> {
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
    kbData: Record<string, unknown>,
  ): Record<string, unknown> {
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
    input: Record<string, unknown>,
  ): Record<string, unknown> {
    const inputData = input?.input || {};
    const inputType = (inputData as Record<string, unknown>)?.type || "TEXT";
    let content = "";
    if (inputType === "TEXT") {
      content = (inputData as Record<string, string>).text || "";
    } else if (inputType === "RETURN_CONTROL") {
      if (
        (inputData as Record<string, unknown>).returnControlResults !==
        undefined
      ) {
        content = JSON.stringify(
          (inputData as Record<string, unknown>).returnControlResults,
        );
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
    observation: Record<string, unknown>,
  ): Record<string, unknown> {
    if (!observation || typeof observation !== "object") return {};
    if ("actionGroupInvocationOutput" in observation) {
      const toolOutput = observation["actionGroupInvocationOutput"] as Record<
        string,
        unknown
      >;
      return getOutputAttributes(toolOutput?.text ?? "");
    }
    if ("codeInterpreterInvocationOutput" in observation) {
      return AttributeExtractor.getAttributesFromCodeInterpreterOutput(
        observation["codeInterpreterInvocationOutput"] as Record<
          string,
          unknown
        >,
      );
    }
    if ("knowledgeBaseLookupOutput" in observation) {
      return AttributeExtractor.getAttributesFromKnowledgeBaseLookupOutput(
        (
          observation["knowledgeBaseLookupOutput"] as
            | { retrievedReferences: Record<string, unknown>[] }
            | undefined
        )?.retrievedReferences ?? [],
      );
    }
    if ("agentCollaboratorInvocationOutput" in observation) {
      return AttributeExtractor.getAttributesFromAgentCollaboratorInvocationOutput(
        observation["agentCollaboratorInvocationOutput"] as Record<
          string,
          unknown
        >,
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
    codeInvocationOutput: Record<string, unknown>,
  ): Record<string, unknown> {
    let outputValue: unknown = null;
    let files: unknown = null;
    if ((codeInvocationOutput as Record<string, unknown>)?.executionOutput) {
      outputValue = (codeInvocationOutput as Record<string, unknown>)
        .executionOutput;
    } else if (
      (codeInvocationOutput as Record<string, unknown>)?.executionError
    ) {
      outputValue = (codeInvocationOutput as Record<string, unknown>)
        .executionError;
    } else if (
      (codeInvocationOutput as Record<string, unknown>)?.executionTimeout
    ) {
      outputValue = "Execution Timeout Error";
    } else if ((codeInvocationOutput as Record<string, unknown>)?.files) {
      files = (codeInvocationOutput as Record<string, unknown>).files;
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
    retrievedReferences: Array<Record<string, unknown>>,
  ): Record<string, unknown> {
    const attributes: Record<string, unknown> = {};
    if (!Array.isArray(retrievedReferences)) return attributes;
    for (let i = 0; i < retrievedReferences.length; i++) {
      const ref: Record<string, unknown> = retrievedReferences[i];
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
    traceData: Record<string, unknown>,
  ): Record<string, unknown> {
    let failureMessage = "";
    if ((traceData as Record<string, unknown>)?.failureCode) {
      failureMessage += `Failure Code: ${(traceData as Record<string, unknown>).failureCode}\n`;
    }
    if ((traceData as Record<string, unknown>)?.failureReason) {
      failureMessage += `Failure Reason: ${(traceData as Record<string, unknown>).failureReason}`;
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
    observation: Record<string, unknown>,
  ): object {
    const events = [
      "actionGroupInvocationOutput",
      "codeInterpreterInvocationOutput",
      "knowledgeBaseLookupOutput",
      "agentCollaboratorInvocationOutput",
    ];
    for (const event of events) {
      if (
        event in observation &&
        (observation as Record<string, unknown>)[event]
      ) {
        return AttributeExtractor.getMetadataAttributes(
          ((
            (observation as Record<string, unknown>)[event] as Record<
              string,
              unknown
            >
          ).metadata as Record<string, unknown>) ?? {},
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
    message: Record<string, unknown>,
    role: string,
  ): import("./types").Message | null {
    if (!message || typeof message !== "object") return null;
    if ((message as Record<string, unknown>).type === "text") {
      return {
        content: ((message as Record<string, unknown>).text as string) || "",
        role,
      };
    }
    if ((message as Record<string, unknown>).type === "tool_use") {
      const toolCallFunction = {
        name: (message as Record<string, string>).name || "",
        arguments: (message as Record<string, string>).input || {},
      };
      const toolCalls = [
        {
          id: (message as Record<string, string>).id || "",
          function: toolCallFunction,
        },
      ];
      return {
        tool_call_id: (message as Record<string, string>).id || "",
        role: "tool",
        tool_calls: toolCalls,
      };
    }
    return {};
  }
}
