import { Span, diag } from "@opentelemetry/api";
import { ConverseCommand } from "@aws-sdk/client-bedrock-runtime";
import {
  SemanticConventions,
  MimeType,
} from "@arizeai/openinference-semantic-conventions";
import { withSafety } from "@arizeai/openinference-core";
import {
  setSpanAttribute,
  aggregateMessages,
  processMessages,
} from "./attribute-helpers";
import {
  SystemPrompt,
  ConverseMessage,
  ConverseRequestBody,
} from "../types/bedrock-types";

/**
 * Type guard to safely validate ConverseRequestBody structure
 */
function isConverseRequestBody(input: unknown): input is ConverseRequestBody {
  if (!input || typeof input !== "object" || input === null) {
    return false;
  }
  
  const obj = input as Record<string, unknown>;
  return (
    "modelId" in obj &&
    typeof obj.modelId === "string" &&
    "messages" in obj &&
    Array.isArray(obj.messages)
  );
}

/**
 * Extracts base request attributes: model, system, provider, parameters
 */
const extractBaseRequestAttributes = withSafety({
  fn: ({ span, input }: { span: Span; input: ConverseRequestBody }): void => {
    if (input.modelId) {
      setSpanAttribute(span, SemanticConventions.LLM_MODEL_NAME, input.modelId);
    }

    setSpanAttribute(span, SemanticConventions.LLM_SYSTEM, "bedrock");

    const inferenceConfig = input.inferenceConfig || {};
    setSpanAttribute(
      span,
      SemanticConventions.LLM_INVOCATION_PARAMETERS,
      JSON.stringify(inferenceConfig),
    );

    setSpanAttribute(
      span,
      SemanticConventions.INPUT_VALUE,
      JSON.stringify(input),
    );
    setSpanAttribute(span, SemanticConventions.INPUT_MIME_TYPE, MimeType.JSON);
  },
  onError: (error) => {
    diag.warn("Error extracting base request attributes:", error);
  },
});

/**
 * Extracts input messages attributes with system prompt aggregation
 */
const extractInputMessagesAttributes = withSafety({
  fn: ({ span, input }: { span: Span; input: ConverseRequestBody }): void => {
    const systemPrompts: SystemPrompt[] = input.system || [];
    const messages: ConverseMessage[] = input.messages || [];

    const aggregatedMessages = aggregateMessages(systemPrompts, messages);

    processMessages({
      span,
      messages: aggregatedMessages,
      baseKey: SemanticConventions.LLM_INPUT_MESSAGES,
    });
  },
  onError: (error) => {
    diag.warn("Error extracting input messages attributes:", error);
  },
});

/**
 * Extracts tool configuration attributes
 */
const extractInputToolAttributes = withSafety({
  fn: ({ span, input }: { span: Span; input: ConverseRequestBody }): void => {
    const toolConfig = input.toolConfig;
    if (!toolConfig?.tools) return;

    toolConfig.tools.forEach((tool, index: number) => {
      if (tool.toolSpec) {
        setSpanAttribute(
          span,
          `${SemanticConventions.LLM_TOOLS}.${index}.${SemanticConventions.TOOL_JSON_SCHEMA}`,
          JSON.stringify(tool.toolSpec.inputSchema?.json || {}),
        );
        setSpanAttribute(
          span,
          `${SemanticConventions.LLM_TOOLS}.${index}.tool.name`,
          tool.toolSpec.name,
        );
        setSpanAttribute(
          span,
          `${SemanticConventions.LLM_TOOLS}.${index}.tool.description`,
          tool.toolSpec.description,
        );
      }
    });
  },
  onError: (error) => {
    diag.warn("Error extracting tool configuration attributes:", error);
  },
});

/**
 * Extracts request attributes from Converse command and sets them on the span
 */
export const extractConverseRequestAttributes = withSafety({
  fn: ({ span, command }: { span: Span; command: ConverseCommand }): void => {
    const input = command.input;
    if (!input || !isConverseRequestBody(input)) {
      return;
    }

    extractBaseRequestAttributes({ span, input });
    extractInputMessagesAttributes({ span, input });
    extractInputToolAttributes({ span, input });
  },
  onError: (error) => {
    diag.warn("Error extracting Converse request attributes:", error);
  },
});
