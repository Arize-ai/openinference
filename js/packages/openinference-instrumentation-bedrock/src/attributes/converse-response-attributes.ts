import { Span, diag } from "@opentelemetry/api";
import {
  SemanticConventions,
  MimeType,
} from "@arizeai/openinference-semantic-conventions";
import {
  withSafety,
  isObjectWithStringKeys,
} from "@arizeai/openinference-core";
import {
  ConverseResponse,
  ConverseOutput,
  TokenUsage,
  Message,
} from "@aws-sdk/client-bedrock-runtime";
import { setSpanAttribute, processMessages } from "./attribute-helpers";
import { isConverseToolUseContent } from "../types/bedrock-types";

/**
 * Type guard to safely validate ConverseResponse structure
 * Ensures the response object has the expected output structure for processing
 *
 * @param response The response object to validate
 * @returns {boolean} True if response is a valid ConverseResponse, false otherwise
 */
function isConverseResponse(response: unknown): response is ConverseResponse {
  if (!isObjectWithStringKeys(response)) {
    return false;
  }

  return (
    "output" in response &&
    typeof response.output === "object" &&
    response.output !== null
  );
}

/**
 * Extracts base response attributes for OpenInference semantic conventions
 * Sets output value, MIME type, and stop reason attributes
 *
 * @param params Object containing extraction parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.response The ConverseResponse to extract attributes from
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

  setSpanAttribute(span, "llm.stop_reason", response.stopReason);
}

/**
 * Extracts output message attributes from Converse response
 * Processes the response message and sets OpenInference message attributes
 *
 * @param params Object containing extraction parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.response The ConverseResponse containing output message
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
 * Identifies and processes tool use blocks in the response message
 *
 * @param params Object containing extraction parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.response The ConverseResponse containing tool calls
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
        JSON.stringify(toolUse.input),
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
 * Extracts token usage statistics with null-safe handling for missing token counts
 * Sets input, output, and total token count attributes following OpenInference conventions
 *
 * @param params Object containing extraction parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.response The ConverseResponse containing usage statistics
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
 * Extracts semantic convention attributes from Converse response and sets them on the span
 * Main entry point for processing Converse API responses with comprehensive error handling
 *
 * Processes:
 * - Base response attributes (output value, MIME type, stop reason)
 * - Output messages with content structure
 * - Tool call attributes from response blocks
 * - Token usage statistics
 *
 * @param params Object containing extraction parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.response The ConverseResponse to extract attributes from
 */
export const extractConverseResponseAttributes = withSafety({
  fn: ({
    span,
    response,
  }: {
    span: Span;
    response: ConverseResponse;
  }): void => {
    if (!isConverseResponse(response)) {
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
