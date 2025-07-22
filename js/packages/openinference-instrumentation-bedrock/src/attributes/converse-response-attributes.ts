import { Span, diag } from "@opentelemetry/api";
import {
  SemanticConventions,
  MimeType,
} from "@arizeai/openinference-semantic-conventions";
import { withSafety } from "@arizeai/openinference-core";
import { setSpanAttribute, processMessages } from "./attribute-helpers";
import {
  ConverseMessage,
  ConverseResponseBody,
  isConverseToolUseContent,
} from "../types/bedrock-types";

/**
 * Type guard to safely validate ConverseResponseBody structure
 */
function isConverseResponseBody(response: unknown): response is ConverseResponseBody {
  if (!response || typeof response !== "object" || response === null) {
    return false;
  }
  
  const obj = response as Record<string, unknown>;
  return (
    "output" in obj &&
    typeof obj.output === "object" &&
    obj.output !== null
  );
}

/**
 * Extracts base response attributes: output value, mime type
 */
const extractBaseResponseAttributes = withSafety({
  fn: (span: Span, response: ConverseResponseBody): void => {
    setSpanAttribute(
      span,
      SemanticConventions.OUTPUT_VALUE,
      JSON.stringify(response),
    );
    setSpanAttribute(span, SemanticConventions.OUTPUT_MIME_TYPE, MimeType.JSON);

    if (response.stopReason) {
      setSpanAttribute(span, "llm.stop_reason", response.stopReason);
    }
  },
  onError: (error) => {
    diag.warn("Error extracting base response attributes:", error);
  },
});

/**
 * Extracts output message attributes from response
 */
const extractOutputMessagesAttributes = withSafety({
  fn: (span: Span, response: ConverseResponseBody): void => {
    const outputMessage = response.output?.message;
    if (!outputMessage) return;

    const message: ConverseMessage = {
      role: outputMessage.role || "assistant",
      content: outputMessage.content || [],
    };

    processMessages(span, [message], SemanticConventions.LLM_OUTPUT_MESSAGES);
  },
  onError: (error) => {
    diag.warn("Error extracting output messages attributes:", error);
  },
});

/**
 * Extracts tool call attributes from response content blocks
 */
const extractToolCallAttributes = withSafety({
  fn: (span: Span, response: ConverseResponseBody): void => {
    const outputMessage = response.output?.message;
    if (!outputMessage?.content) return;

    const toolUseBlocks = outputMessage.content.filter(isConverseToolUseContent);

    toolUseBlocks.forEach((content, toolCallIndex: number) => {
      const toolUse = content.toolUse;
      if (toolUse) {
        setSpanAttribute(
          span,
          `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`,
          toolUse.name,
        );
        setSpanAttribute(
          span,
          `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`,
          toolUse.input ? JSON.stringify(toolUse.input) : undefined,
        );
        setSpanAttribute(
          span,
          `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_ID}`,
          toolUse.toolUseId,
        );
      }
    });
  },
  onError: (error) => {
    diag.warn("Error extracting tool call attributes:", error);
  },
});

/**
 * Extracts usage statistics with null-safe handling for missing token counts
 */
const extractUsageAttributes = withSafety({
  fn: (span: Span, response: ConverseResponseBody): void => {
    const usage = response.usage;
    if (!usage) return;

    setSpanAttribute(
      span,
      SemanticConventions.LLM_TOKEN_COUNT_PROMPT,
      usage.inputTokens,
    );
    setSpanAttribute(
      span,
      SemanticConventions.LLM_TOKEN_COUNT_COMPLETION,
      usage.outputTokens,
    );
    setSpanAttribute(
      span,
      SemanticConventions.LLM_TOKEN_COUNT_TOTAL,
      usage.totalTokens,
    );
  },
  onError: (error) => {
    diag.warn("Error extracting usage attributes:", error);
  },
});

/**
 * Extracts response attributes from Converse response and sets them on the span
 */
export const extractConverseResponseAttributes = withSafety({
  fn: (span: Span, response: ConverseResponseBody): void => {
    if (!response || !isConverseResponseBody(response)) {
      return;
    }

    extractBaseResponseAttributes(span, response);
    extractOutputMessagesAttributes(span, response);
    extractToolCallAttributes(span, response);
    extractUsageAttributes(span, response);
  },
  onError: (error) => {
    diag.warn("Error extracting Converse response attributes:", error);
  },
});
