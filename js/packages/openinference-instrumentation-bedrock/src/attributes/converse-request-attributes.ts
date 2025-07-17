import { Span } from "@opentelemetry/api";
import { ConverseCommand } from "@aws-sdk/client-bedrock-runtime";
import { SemanticConventions, MimeType } from "@arizeai/openinference-semantic-conventions";
import { 
  setSpanAttribute, 
  aggregateMessages, 
  processMessages 
} from "./attribute-helpers";
import { 
  ConverseRequestBody, 
  SystemPrompt, 
  ConverseMessage 
} from "../types/bedrock-types";

/**
 * Extracts request attributes from Converse command and sets them on the span
 * Following Python implementation patterns for incremental attribute building
 */
export function extractConverseRequestAttributes(span: Span, command: ConverseCommand): void {
  const input = command.input;
  if (!input) return;

  // Extract base request attributes
  extractBaseRequestAttributes(span, input);

  // Extract input messages attributes  
  extractInputMessagesAttributes(span, input);

  // Extract tool configuration attributes
  extractInputToolAttributes(span, input);
}

/**
 * Extracts base request attributes: model, system, provider, parameters
 * Following Python's incremental attribute building pattern
 */
function extractBaseRequestAttributes(span: Span, input: any): void {
  // Model identification
  if (input.modelId) {
    setSpanAttribute(span, SemanticConventions.LLM_MODEL_NAME, input.modelId);
  }

  // System attribute for provider identification
  setSpanAttribute(span, SemanticConventions.LLM_SYSTEM, "bedrock");

  // Inference configuration as invocation parameters
  if (input.inferenceConfig) {
    setSpanAttribute(
      span, 
      SemanticConventions.LLM_INVOCATION_PARAMETERS, 
      JSON.stringify(input.inferenceConfig)
    );
  }

  // Set input value as JSON
  setSpanAttribute(span, SemanticConventions.INPUT_VALUE, JSON.stringify(input));
  setSpanAttribute(span, SemanticConventions.INPUT_MIME_TYPE, MimeType.JSON);
}

/**
 * Extracts input messages attributes with system prompt aggregation
 * Following Python's message processing patterns
 */
function extractInputMessagesAttributes(span: Span, input: any): void {
  const systemPrompts: SystemPrompt[] = input.system || [];
  const messages: ConverseMessage[] = input.messages || [];

  // Aggregate system prompts and messages following Python pattern
  const aggregatedMessages = aggregateMessages(systemPrompts, messages);

  // Process all messages with proper indexing
  processMessages(span, aggregatedMessages, SemanticConventions.LLM_INPUT_MESSAGES);
}

/**
 * Extracts tool configuration attributes
 * Following Python's tool processing patterns
 */
function extractInputToolAttributes(span: Span, input: any): void {
  const toolConfig = input.toolConfig;
  if (!toolConfig?.tools) return;

  // Process each tool definition
  toolConfig.tools.forEach((tool: any, index: number) => {
    if (tool.toolSpec) {
      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_TOOLS}.${index}.${SemanticConventions.TOOL_JSON_SCHEMA}`,
        JSON.stringify(tool.toolSpec.inputSchema?.json || {})
      );
      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_TOOLS}.${index}.tool.name`,
        tool.toolSpec.name
      );
      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_TOOLS}.${index}.tool.description`,
        tool.toolSpec.description
      );
    }
  });
}