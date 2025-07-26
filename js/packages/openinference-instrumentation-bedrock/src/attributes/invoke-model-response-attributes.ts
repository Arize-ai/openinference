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
import { withSafety } from "@arizeai/openinference-core";
import {
  InvokeModelResponseBody,
  TextContent,
  isTextContent,
  isToolUseContent,
} from "../types/bedrock-types";
import { setSpanAttribute } from "./attribute-helpers";


/**
 * Safely parses the InvokeModel response body with comprehensive error handling
 * Handles multiple response body formats and provides null fallback on error
 *
 * @param response The InvokeModelResponse containing the response body to parse
 * @returns {InvokeModelResponseBody | null} Parsed response body or null on error
 */
const parseResponseBody = withSafety({
  fn: (response: InvokeModelResponse): InvokeModelResponseBody => {
    if (!response.body) {
      throw new Error("Response body is missing");
    }

    let responseText: string;
    if (typeof response.body === "string") {
      responseText = response.body;
    } else if (response.body instanceof Uint8Array) {
      responseText = new TextDecoder().decode(response.body);
    } else {
      // Handle other potential types
      responseText = new TextDecoder().decode(response.body as Uint8Array);
    }

    return JSON.parse(responseText) as InvokeModelResponseBody;
  },
  onError: (error) => {
    diag.warn("Error parsing response body:", error);
    return null;
  },
});

/**
 * Type guard to check if response body contains a simple single text content
 * Combines all checks needed to safely access the text content without casting
 * 
 * @param responseBody The response body to check
 * @returns {boolean} True if response contains a single text content block
 */
function isSimpleTextResponse(
  responseBody: InvokeModelResponseBody,
): responseBody is InvokeModelResponseBody & {
  content: [TextContent];
} {
  return (
    responseBody.content &&
    Array.isArray(responseBody.content) &&
    responseBody.content.length === 1 &&
    isTextContent(responseBody.content[0])
  );
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
function extractOutputMessagesAttributes({
  responseBody,
  span,
}: {
  responseBody: InvokeModelResponseBody;
  span: Span;
}): void {
  // Extract full response body as primary output value
  const outputValue = JSON.stringify(responseBody);
  setSpanAttribute(span, SemanticConventions.OUTPUT_VALUE, outputValue);
  setSpanAttribute(span, SemanticConventions.OUTPUT_MIME_TYPE, MimeType.JSON);

  // Determine if this is a simple text response or complex multimodal response
  if (isSimpleTextResponse(responseBody)) {
    // For simple text responses, use the message content
    // No cast needed - type guard ensures content[0] is TextContent
    const textBlock = responseBody.content[0];
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`,
      "assistant",
    );
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`,
      textBlock.text,
    );
  } else {
    // For complex multimodal responses, use the detailed message structure
    addOutputMessageContentAttributes({ responseBody, span });
  }
}

/**
 * Extracts tool call attributes from InvokeModel response body
 * Identifies and processes tool use blocks in the response content
 *
 * @param params Object containing extraction parameters
 * @param params.responseBody The parsed response body containing tool calls
 * @param params.span The OpenTelemetry span to set attributes on
 */
function extractToolCallsAttributes({
  responseBody,
  span,
}: {
  responseBody: InvokeModelResponseBody;
  span: Span;
}): void {
  if (!responseBody.content || !Array.isArray(responseBody.content)) {
    return;
  }

  const toolUseBlocks = responseBody.content.filter(isToolUseContent);

  toolUseBlocks.forEach((content, toolCallIndex) => {
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
  });
}

/**
 * Extracts token usage attributes from InvokeModel response body
 * Processes standard and cache-related token counts following OpenInference conventions
 *
 * @param params Object containing extraction parameters
 * @param params.responseBody The parsed response body containing usage statistics
 * @param params.span The OpenTelemetry span to set attributes on
 */
function extractUsageAttributes({
  responseBody,
  span,
}: {
  responseBody: InvokeModelResponseBody;
  span: Span;
}): void {
  if (!responseBody.usage) {
    return;
  }

  const usage = responseBody.usage;

  // Standard token counts
  setSpanAttribute(
    span,
    SemanticConventions.LLM_TOKEN_COUNT_PROMPT,
    usage.input_tokens,
  );
  setSpanAttribute(
    span,
    SemanticConventions.LLM_TOKEN_COUNT_COMPLETION,
    usage.output_tokens,
  );

  // Cache-related token attributes (if present)
  setSpanAttribute(
    span,
    SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
    usage.cache_read_input_tokens,
  );
  setSpanAttribute(
    span,
    SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
    usage.cache_creation_input_tokens,
  );
}

/**
 * Adds detailed output message content structure attributes for multi-modal responses
 * Processes individual content blocks and sets structured message attributes
 *
 * @param params Object containing extraction parameters
 * @param params.responseBody The parsed response body containing structured content
 * @param params.span The OpenTelemetry span to set attributes on
 */
function addOutputMessageContentAttributes({
  responseBody,
  span,
}: {
  responseBody: InvokeModelResponseBody;
  span: Span;
}): void {
  if (!responseBody.content || !Array.isArray(responseBody.content)) {
    return;
  }

  // Process each content block in the response
  responseBody.content.forEach((content, contentIndex) => {
    const contentPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}`;

    if (isTextContent(content)) {
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
    }
  });
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
  }: {
    span: Span;
    response: InvokeModelResponse;
  }): void => {
    if (!response.body) {
      return;
    }

    const responseBody = parseResponseBody(response);
    if (!responseBody) {
      return;
    }

    // Extract all response attributes using named parameters
    extractOutputMessagesAttributes({ responseBody, span });
    extractToolCallsAttributes({ responseBody, span });
    extractUsageAttributes({ responseBody, span });
  },
  onError: (error) => {
    diag.warn("Error extracting InvokeModel response attributes:", error);
  },
});
