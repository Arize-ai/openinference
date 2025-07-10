/**
 * Response attribute extraction for AWS Bedrock instrumentation
 *
 * Handles extraction of semantic convention attributes from InvokeModel responses including:
 * - Output message processing
 * - Tool call extraction
 * - Usage/token count processing
 */

import { Span, diag } from "@opentelemetry/api";
import {
  SemanticConventions,
  MimeType,
} from "@arizeai/openinference-semantic-conventions";
import { InvokeModelResponse } from "@aws-sdk/client-bedrock-runtime";
import {
  InvokeModelResponseBody,
  isTextContent,
  isToolUseContent,
} from "../types/bedrock-types";
import { TextDecoder } from "util";

/**
 * Extracts output messages attributes from response body
 */
export function extractOutputMessagesAttributes(
  responseBody: InvokeModelResponseBody,
  span: Span,
): void {
  // Extract assistant's message text as primary output value
  const outputValue = extractPrimaryOutputValue(responseBody);

  span.setAttributes({
    [SemanticConventions.OUTPUT_VALUE]: outputValue,
    [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
  });

  // Add structured output message attributes for text content
  if (responseBody.content && Array.isArray(responseBody.content)) {
    const textBlocks = responseBody.content.filter(isTextContent);
    textBlocks.forEach((content) => {
      span.setAttributes({
        [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
          "assistant",
        [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
          content.text,
      });
    });
  }
}

/**
 * Extracts tool call attributes from response body
 */
export function extractToolCallAttributes(
  responseBody: InvokeModelResponseBody,
  span: Span,
): void {
  if (!responseBody.content || !Array.isArray(responseBody.content)) {
    return;
  }

  let toolCallIndex = 0;
  const toolUseBlocks = responseBody.content.filter(isToolUseContent);

  toolUseBlocks.forEach((content) => {
    // Extract tool call attributes following OpenAI pattern
    const toolCallAttributes: Record<string, string> = {};

    if (content.name) {
      toolCallAttributes[
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
      ] = content.name;
    }
    if (content.input) {
      toolCallAttributes[
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
      ] = JSON.stringify(content.input);
    }
    if (content.id) {
      toolCallAttributes[
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_ID}`
      ] = content.id;
    }

    span.setAttributes(toolCallAttributes);
    toolCallIndex++;
  });
}

/**
 * Extracts usage attributes from response body
 */
export function extractUsageAttributes(
  responseBody: InvokeModelResponseBody,
  span: Span,
): void {
  // Add token usage metrics
  if (!responseBody.usage) {
    return;
  }

  const tokenAttributes: Record<string, number> = {};

  if (responseBody.usage.input_tokens) {
    tokenAttributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT] =
      responseBody.usage.input_tokens;
  }
  if (responseBody.usage.output_tokens) {
    tokenAttributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION] =
      responseBody.usage.output_tokens;
  }
  if (responseBody.usage.input_tokens && responseBody.usage.output_tokens) {
    tokenAttributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL] =
      responseBody.usage.input_tokens + responseBody.usage.output_tokens;
  }

  span.setAttributes(tokenAttributes);
}

/**
 * Extracts semantic convention attributes from InvokeModel response and adds them to the span
 */
export function extractInvokeModelResponseAttributes(
  span: Span,
  response: InvokeModelResponse,
): void {
  try {
    if (!response.body) return;

    const responseBody = parseResponseBody(response);

    // Extract output messages attributes
    extractOutputMessagesAttributes(responseBody, span);

    // Extract tool call attributes
    extractToolCallAttributes(responseBody, span);

    // Extract usage attributes
    extractUsageAttributes(responseBody, span);
  } catch (error) {
    diag.warn("Failed to extract InvokeModel response attributes:", error);
  }
}

// Helper functions

/**
 * Safely parses the response body
 */
function parseResponseBody(
  response: InvokeModelResponse,
): InvokeModelResponseBody {
  if (!response.body) {
    throw new Error("Response body is missing");
  }
  const responseText = new TextDecoder().decode(response.body);
  return JSON.parse(responseText);
}

/**
 * Extracts the primary output value from response content
 */
function extractPrimaryOutputValue(
  responseBody: InvokeModelResponseBody,
): string {
  if (!responseBody.content || !Array.isArray(responseBody.content)) {
    return "";
  }

  const textContent = responseBody.content.find(isTextContent);
  return textContent?.text || "";
}
