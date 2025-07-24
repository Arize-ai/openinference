import { Span, diag } from "@opentelemetry/api";
import {
  SemanticConventions,
  MimeType,
} from "@arizeai/openinference-semantic-conventions";
import { withSafety } from "@arizeai/openinference-core";
import { 
  ConverseResponse,
  ConverseOutput,
  TokenUsage,
  Message,
} from "@aws-sdk/client-bedrock-runtime";
import { setSpanAttribute, processMessages } from "./attribute-helpers";
import {
  isConverseToolUseContent,
} from "../types/bedrock-types";

/**
 * Type guard to safely validate ConverseResponse structure
 */
function isConverseResponse(response: unknown): response is ConverseResponse {
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
function extractBaseResponseAttributes({
  span,
  response,
}: {
  span: Span;
  response: ConverseResponse;
}): void {
  setSpanAttribute(
    span,
    SemanticConventions.OUTPUT_VALUE,
    JSON.stringify(response),
  );
  setSpanAttribute(span, SemanticConventions.OUTPUT_MIME_TYPE, MimeType.JSON);

  if (response.stopReason) {
    setSpanAttribute(span, "llm.stop_reason", response.stopReason);
  }
}

/**
 * Extracts output message attributes from response
 */
function extractOutputMessagesAttributes({
  span,
  response,
}: {
  span: Span;
  response: ConverseResponse;
}): void {
  const output: ConverseOutput | undefined = response.output;
  if (!output || !("message" in output)) return;

  const outputMessage: Message | undefined = output.message;
  if (!outputMessage) return;

  processMessages({
    span,
    messages: [outputMessage],
    baseKey: SemanticConventions.LLM_OUTPUT_MESSAGES,
  });
}

/**
 * Extracts tool call attributes from response content blocks
 */
function extractToolCallAttributes({
  span,
  response,
}: {
  span: Span;
  response: ConverseResponse;
}): void {
  const output: ConverseOutput | undefined = response.output;
  if (!output || !("message" in output)) return;

  const outputMessage: Message | undefined = output.message;
  if (!outputMessage || !outputMessage.content) return;

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
}

/**
 * Extracts usage statistics with null-safe handling for missing token counts
 */
function extractUsageAttributes({
  span,
  response,
}: {
  span: Span;
  response: ConverseResponse;
}): void {
  const usage: TokenUsage | undefined = response.usage;
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
}

/**
 * Extracts response attributes from Converse response and sets them on the span
 */
export const extractConverseResponseAttributes = withSafety({
  fn: ({ span, response }: { span: Span; response: ConverseResponse }): void => {
    if (!response || !isConverseResponse(response)) {
      return;
    }

    extractBaseResponseAttributes({ span, response });
    extractOutputMessagesAttributes({ span, response });
    extractToolCallAttributes({ span, response });
    extractUsageAttributes({ span, response });
  },
  onError: (error) => {
    diag.warn("Error extracting Converse response attributes:", error);
  },
});
