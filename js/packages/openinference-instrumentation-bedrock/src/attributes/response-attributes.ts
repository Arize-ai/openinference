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
import { setSpanAttribute } from "./attribute-helpers";

/**
 * Extracts output messages attributes from response body
 */
export function extractOutputMessagesAttributes(
  responseBody: InvokeModelResponseBody,
  span: Span,
): void {
  // Extract full response body as primary output value
  const outputValue = extractPrimaryOutputValue(responseBody);

  // Use JSON mime type for full response body
  const mimeType = MimeType.JSON;

  setSpanAttribute(span, SemanticConventions.OUTPUT_VALUE, outputValue);
  setSpanAttribute(span, SemanticConventions.OUTPUT_MIME_TYPE, mimeType);

  // Determine if this is a simple text response or complex multimodal response
  const isSimpleTextResponse = responseBody.content && 
    Array.isArray(responseBody.content) && 
    responseBody.content.length === 1 && 
    isTextContent(responseBody.content[0]);

  if (isSimpleTextResponse) {
    // For simple text responses, use only the top-level message.content
    const textBlock = responseBody.content[0];
    if (isTextContent(textBlock)) {
      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`,
        "assistant"
      );
      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`,
        textBlock.text
      );
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
    // Extract tool call attributes
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`,
      content.name
    );
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`,
      content.input ? JSON.stringify(content.input) : undefined
    );
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_ID}`,
      content.id
    );

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

  setSpanAttribute(span, SemanticConventions.LLM_TOKEN_COUNT_PROMPT, responseBody.usage.input_tokens);
  setSpanAttribute(span, SemanticConventions.LLM_TOKEN_COUNT_COMPLETION, responseBody.usage.output_tokens);
  
  // Note: Don't calculate total, only set what's in response
  // If the response includes total tokens, we could add:
  // setSpanAttribute(span, SemanticConventions.LLM_TOKEN_COUNT_TOTAL, responseBody.usage.total_tokens);

  // Add cache-related token attributes
  setSpanAttribute(span, `${SemanticConventions.LLM_TOKEN_COUNT_PROMPT}.cache_read`, responseBody.usage.cache_read_input_tokens);
  setSpanAttribute(span, `${SemanticConventions.LLM_TOKEN_COUNT_PROMPT}.cache_write`, responseBody.usage.cache_creation_input_tokens);
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
      setSpanAttribute(span, `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`, "text");
      setSpanAttribute(span, `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`, content.text);
    } else if (isToolUseContent(content)) {
      setSpanAttribute(span, `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`, "tool_use");
      setSpanAttribute(span, `${contentPrefix}.message_content.tool_use.name`, content.name);
      setSpanAttribute(span, `${contentPrefix}.message_content.tool_use.input`, JSON.stringify(content.input));
      setSpanAttribute(span, `${contentPrefix}.message_content.tool_use.id`, content.id);
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
  // Use full response body as JSON
  return JSON.stringify(responseBody);
}
