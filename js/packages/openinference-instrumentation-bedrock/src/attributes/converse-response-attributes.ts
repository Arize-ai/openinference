import { Span } from "@opentelemetry/api";
import { SemanticConventions, MimeType } from "@arizeai/openinference-semantic-conventions";
import { 
  setSpanAttribute, 
  processMessages 
} from "./attribute-helpers";
import { 
  ConverseResponseBody,
  ConverseMessage,
  isConverseToolUseContent
} from "../types/bedrock-types";

/**
 * Extracts response attributes from Converse response and sets them on the span
 * Following Python implementation patterns for incremental attribute building
 */
export function extractConverseResponseAttributes(span: Span, response: any): void {
  if (!response) return;

  // Extract base response attributes
  extractBaseResponseAttributes(span, response);

  // Extract output messages attributes
  extractOutputMessagesAttributes(span, response);

  // Extract tool call attributes from response content
  extractToolCallAttributes(span, response);

  // Extract usage statistics
  extractUsageAttributes(span, response);
}

/**
 * Extracts base response attributes: output value, mime type
 * Following Python's incremental attribute building pattern
 */
function extractBaseResponseAttributes(span: Span, response: any): void {
  // Set output value as JSON
  setSpanAttribute(span, SemanticConventions.OUTPUT_VALUE, JSON.stringify(response));
  setSpanAttribute(span, SemanticConventions.OUTPUT_MIME_TYPE, MimeType.JSON);

  // Stop reason if available
  if (response.stopReason) {
    setSpanAttribute(span, "llm.stop_reason", response.stopReason);
  }
}

/**
 * Extracts output message attributes from response
 * Following Python's message processing patterns
 */
function extractOutputMessagesAttributes(span: Span, response: any): void {
  const outputMessage = response.output?.message;
  if (!outputMessage) return;

  // Convert to our message format for processing
  const message: ConverseMessage = {
    role: outputMessage.role || "assistant",
    content: outputMessage.content || []
  };

  // Process the single output message (index 0)
  processMessages(span, [message], SemanticConventions.LLM_OUTPUT_MESSAGES);
}

/**
 * Extracts tool call attributes from response content blocks
 * Following Python's tool call processing patterns
 */
function extractToolCallAttributes(span: Span, response: any): void {
  const outputMessage = response.output?.message;
  if (!outputMessage?.content) return;

  // Find tool use content blocks
  const toolUseBlocks = outputMessage.content.filter(isConverseToolUseContent);
  
  toolUseBlocks.forEach((content: any, toolCallIndex: number) => {
    const toolUse = content.toolUse;
    if (toolUse) {
      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`,
        toolUse.name
      );
      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`,
        toolUse.input ? JSON.stringify(toolUse.input) : undefined
      );
      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_ID}`,
        toolUse.toolUseId
      );
    }
  });
}

/**
 * Extracts usage statistics with null-safe handling for missing token counts
 * Following Python's token count extraction patterns
 */
function extractUsageAttributes(span: Span, response: any): void {
  const usage = response.usage;
  if (!usage) return;

  // Set only token counts that are available (null-safe)
  setSpanAttribute(span, SemanticConventions.LLM_TOKEN_COUNT_PROMPT, usage.inputTokens);
  setSpanAttribute(span, SemanticConventions.LLM_TOKEN_COUNT_COMPLETION, usage.outputTokens);
  setSpanAttribute(span, SemanticConventions.LLM_TOKEN_COUNT_TOTAL, usage.totalTokens);
}