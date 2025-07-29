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
  LLMSystem,
} from "@arizeai/openinference-semantic-conventions";
import { InvokeModelResponse } from "@aws-sdk/client-bedrock-runtime";
import { withSafety } from "@arizeai/openinference-core";
import {
  normalizeResponseContentBlocks,
  isSimpleTextResponse,
  parseResponseBody,
  normalizeUsageAttributes,
} from "./invoke-model-helpers";
import {
  BedrockMessage,
  isTextContent,
  isToolUseContent,
} from "../types/bedrock-types";
import { setSpanAttribute } from "./attribute-helpers";

/**
 * Processes message content and extracts both text content and tool call attributes
 * Handles multi-modal responses by iterating through content blocks and processing
 * text content and tool use blocks with proper indexing
 *
 * @param params Object containing extraction parameters
 * @param params.message The normalized BedrockMessage containing structured content
 * @param params.span The OpenTelemetry span to set attributes on
 */
function addOutputMessageContentAttributes({
  message,
  span,
}: {
  message: BedrockMessage;
  span: Span;
}): void {
  if (!Array.isArray(message.content)) {
    return;
  }

  // Set the message role
  setSpanAttribute(
    span,
    `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`,
    message.role,
  );

  // Use separate counters for different content types
  let contentBlockIndex = 0;
  let toolCallIndex = 0;

  // Process each content block in the message
  message.content.forEach((content) => {
    if (isTextContent(content)) {
      // Handle text content with sequential content indexing
      const contentPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENTS}.${contentBlockIndex}`;

      setSpanAttribute(
        span,
        `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`,
        "text",
      );
      setSpanAttribute(
        span,
        `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`,
        content.text,
      );

      contentBlockIndex++;
    } else if (isToolUseContent(content)) {
      // Handle tool use content with sequential tool call indexing
      const toolCallPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}`;

      setSpanAttribute(
        span,
        `${toolCallPrefix}.${SemanticConventions.TOOL_CALL_ID}`,
        content.id,
      );
      setSpanAttribute(
        span,
        `${toolCallPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`,
        content.name,
      );
      setSpanAttribute(
        span,
        `${toolCallPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`,
        JSON.stringify(content.input),
      );

      toolCallIndex++;
    }
  });
}

/**
 * Extracts output messages attributes from InvokeModel response body
 * Processes response content and sets OpenInference output message attributes
 * Handles both simple text responses and complex multi-modal responses
 *
 * @param params Object containing extraction parameters
 * @param params.responseBody The parsed response body containing content and metadata
 * @param params.span The OpenTelemetry span to set attributes on
 */
const extractOutputMessagesAttributes = withSafety({
  fn: ({
    normalizedMessage,
    span,
  }: {
    normalizedMessage: BedrockMessage;
    span: Span;
  }) => {
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`,
      normalizedMessage.role,
    );

    // Determine if this is a simple text response or complex multimodal response
    if (isSimpleTextResponse(normalizedMessage)) {
      // For simple text responses, use the message content
      // No cast needed - type guard ensures content[0] is TextContent
      const textBlock = normalizedMessage.content[0];

      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`,
        textBlock.text,
      );
    } else {
      // For complex multimodal responses, use the detailed message structure
      addOutputMessageContentAttributes({ message: normalizedMessage, span });
    }
  },
  onError: (error) => {
    diag.warn("Error extracting output messages attributes:", error);
  },
});

/**
 * Extracts token usage attributes from InvokeModel response body
 * Uses normalized usage attributes to handle different provider formats
 * Processes standard and cache-related token counts following OpenInference conventions
 *
 * @param params Object containing extraction parameters
 * @param params.responseBody The parsed response body containing usage statistics
 * @param params.modelType The LLM system type to determine extraction strategy
 * @param params.span The OpenTelemetry span to set attributes on
 */
function extractUsageAttributes({
  responseBody,
  modelType,
  span,
}: {
  responseBody: Record<string, unknown>;
  modelType: LLMSystem;
  span: Span;
}): void {
  const usage = normalizeUsageAttributes(responseBody, modelType);
  if (!usage) {
    return;
  }

  // Token counts - only set if defined
  if (usage.input_tokens !== undefined) {
    setSpanAttribute(
      span,
      SemanticConventions.LLM_TOKEN_COUNT_PROMPT,
      usage.input_tokens,
    );
  }
  if (usage.output_tokens !== undefined) {
    setSpanAttribute(
      span,
      SemanticConventions.LLM_TOKEN_COUNT_COMPLETION,
      usage.output_tokens,
    );
  }

  if (usage.total_tokens !== undefined) {
    setSpanAttribute(
      span,
      SemanticConventions.LLM_TOKEN_COUNT_TOTAL,
      usage.total_tokens,
    );
  }

  if (usage.cache_read_input_tokens !== undefined) {
    setSpanAttribute(
      span,
      SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
      usage.cache_read_input_tokens,
    );
  }
  if (usage.cache_creation_input_tokens !== undefined) {
    setSpanAttribute(
      span,
      SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
      usage.cache_creation_input_tokens,
    );
  }
}

/**
 * Extracts semantic convention attributes from InvokeModel response and sets them on the span
 * Main entry point for processing InvokeModel API responses with comprehensive error handling
 *
 * Processes:
 * - Response body parsing from multiple formats
 * - Output messages with content structure
 * - Tool call attributes from response blocks
 * - Token usage statistics including cache metrics
 *
 * @param params Object containing extraction parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.response The InvokeModelResponse to extract attributes from
 */
export const extractInvokeModelResponseAttributes = withSafety({
  fn: ({
    span,
    response,
    modelType,
  }: {
    span: Span;
    response: InvokeModelResponse;
    modelType: LLMSystem;
  }): void => {
    if (!response.body) {
      return;
    }

    const responseBody = parseResponseBody(response);
    if (!responseBody) {
      return;
    }
    //extract full response body as primary output
    const outputValue = JSON.stringify(responseBody);
    setSpanAttribute(span, SemanticConventions.OUTPUT_VALUE, outputValue);
    setSpanAttribute(
      span,
      SemanticConventions.OUTPUT_MIME_TYPE,
      "application/json",
    );

    // Normalize the response body to a standard BedrockMessage format
    const normalizedMessage = normalizeResponseContentBlocks(
      responseBody,
      modelType,
    );
    if (normalizedMessage) {
      // Extract all response attributes using named parameters
      extractOutputMessagesAttributes({ normalizedMessage, span });
    }

    extractUsageAttributes({ responseBody, modelType, span });
  },
  onError: (error) => {
    diag.warn("Error extracting InvokeModel response attributes:", error);
  },
});
