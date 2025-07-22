import { Span } from "@opentelemetry/api";
import { ConverseCommand } from "@aws-sdk/client-bedrock-runtime";
import {
  SemanticConventions,
  MimeType,
} from "@arizeai/openinference-semantic-conventions";
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
 * Extracts request attributes from Converse command and sets them on the span
 */
export function extractConverseRequestAttributes(
  span: Span,
  command: ConverseCommand,
): void {
  const input = command.input as ConverseRequestBody;
  if (!input) return;

  extractBaseRequestAttributes(span, input);
  extractInputMessagesAttributes(span, input);
  extractInputToolAttributes(span, input);
}

/**
 * Extracts base request attributes: model, system, provider, parameters
 */
function extractBaseRequestAttributes(
  span: Span,
  input: ConverseRequestBody,
): void {
  // Model identification
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
}

/**
 * Extracts input messages attributes with system prompt aggregation
 */
function extractInputMessagesAttributes(
  span: Span,
  input: ConverseRequestBody,
): void {
  const systemPrompts: SystemPrompt[] = input.system || [];
  const messages: ConverseMessage[] = input.messages || [];

  const aggregatedMessages = aggregateMessages(systemPrompts, messages);

  processMessages(
    span,
    aggregatedMessages,
    SemanticConventions.LLM_INPUT_MESSAGES,
  );
}

/**
 * Extracts tool configuration attributes
 */
function extractInputToolAttributes(
  span: Span,
  input: ConverseRequestBody,
): void {
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
}
