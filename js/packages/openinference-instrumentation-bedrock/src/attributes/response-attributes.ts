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
  // Extract full response body as primary output value
  const outputValue = extractPrimaryOutputValue(responseBody);

  // Use JSON mime type for full response body (following OpenAI/LangChain pattern)
  const mimeType = MimeType.JSON;

  span.setAttributes({
    [SemanticConventions.OUTPUT_VALUE]: outputValue,
    [SemanticConventions.OUTPUT_MIME_TYPE]: mimeType,
  });

  // Determine if this is a simple text response or complex multimodal response
  const isSimpleTextResponse = responseBody.content && 
    Array.isArray(responseBody.content) && 
    responseBody.content.length === 1 && 
    isTextContent(responseBody.content[0]);

  if (isSimpleTextResponse) {
    // For simple text responses, use only the top-level message.content
    const textBlock = responseBody.content[0];
    if (isTextContent(textBlock)) {
      span.setAttributes({
        [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
          "assistant",
        [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
          textBlock.text,
      });
    }
  } else {
    // For complex multimodal responses, use only the detailed message.contents structure
    addOutputMessageContentAttributes(responseBody, span);
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
  // Note: Following OpenAI pattern - don't calculate total, only set what's in response
  // If the response includes total tokens, we could add:
  // if (responseBody.usage.total_tokens) {
  //   tokenAttributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL] = responseBody.usage.total_tokens;
  // }

  // Add cache-related token attributes
  if (responseBody.usage.cache_read_input_tokens !== undefined) {
    tokenAttributes[`${SemanticConventions.LLM_TOKEN_COUNT_PROMPT}.cache_read`] =
      responseBody.usage.cache_read_input_tokens;
  }
  if (responseBody.usage.cache_creation_input_tokens !== undefined) {
    tokenAttributes[`${SemanticConventions.LLM_TOKEN_COUNT_PROMPT}.cache_write`] =
      responseBody.usage.cache_creation_input_tokens;
  }

  span.setAttributes(tokenAttributes);
}

/**
 * Adds detailed output message content structure attributes
 * Phase 2: Message Content Structure Enhancement
 */
function addOutputMessageContentAttributes(
  responseBody: InvokeModelResponseBody,
  span: Span,
): void {
  if (!responseBody.content || !Array.isArray(responseBody.content)) {
    return;
  }

  // Process each content block in the response
  responseBody.content.forEach((content, contentIndex) => {
    const contentPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}`;
    
    if (isTextContent(content)) {
      span.setAttributes({
        [`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`]: "text",
        [`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`]: content.text,
      });
    } else if (isToolUseContent(content)) {
      span.setAttributes({
        [`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`]: "tool_use",
        [`${contentPrefix}.message_content.tool_use.name`]: content.name,
        [`${contentPrefix}.message_content.tool_use.input`]: JSON.stringify(content.input),
        [`${contentPrefix}.message_content.tool_use.id`]: content.id,
      });
    }
  });
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
 * Extracts the primary output value as full response body JSON
 */
function extractPrimaryOutputValue(
  responseBody: InvokeModelResponseBody,
): string {
  // Following OpenAI and LangChain pattern: use full response body as JSON
  return JSON.stringify(responseBody);
}
