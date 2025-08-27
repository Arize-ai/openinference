/**
 * Request attribute extraction for AWS Bedrock instrumentation
 *
 * Handles extraction of semantic convention attributes from InvokeModel requests including:
 * - Base model and system attributes
 * - Input message processing
 * - Tool definition processing
 * - Invocation parameters
 */

import { Span, diag } from "@opentelemetry/api";
import {
  SemanticConventions,
  MimeType,
  LLMSystem,
} from "@arizeai/openinference-semantic-conventions";
import { InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import {
  withSafety,
  isObjectWithStringKeys,
} from "@arizeai/openinference-core";
import {
  InvokeModelRequestBody,
  BedrockMessage,
  isTextContent,
  isImageContent,
  isToolUseContent,
} from "../types/bedrock-types";
import { setSpanAttribute, extractModelName } from "./attribute-helpers";
import {
  extractInvocationParameters,
  parseRequestBody,
  extractToolResultBlocks,
  formatImageUrl,
  normalizeRequestContentBlocks,
} from "./invoke-model-helpers";

// Helper functions
/**
 * Adds message attributes to the span following OpenInference conventions
 * Processes role, content, tool calls, and tool results with proper indexing
 *
 * @param params Object containing message processing parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.message The Bedrock message to process
 * @param params.index The message index in the conversation
 */
function addMessageAttributes({
  span,
  message,
  index,
}: {
  span: Span;
  message: BedrockMessage;
  index: number;
}): void {
  if (!message.role) return;

  setSpanAttribute(
    span,
    `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_ROLE}`,
    message.role,
  );

  if (typeof message.content === "string") {
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_CONTENT}`,
      message.content,
    );
  } else {
    addMessageContentAttributes({ span, message, messageIndex: index });
  }

  // Handle tool calls and results at the message level
  if (Array.isArray(message.content)) {
    handleToolCallsInMessage({ span, message, messageIndex: index });
    handleToolResultsInMessage({ span, message, messageIndex: index });
  }
}

/**
 * Handles tool calls within a message using OpenInference semantic conventions
 * Extracts tool call information and sets appropriate attributes with indexing
 *
 * @param params Object containing tool call processing parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.message The Bedrock message containing tool calls
 * @param params.messageIndex The message index in the conversation
 */
function handleToolCallsInMessage({
  span,
  message,
  messageIndex,
}: {
  span: Span;
  message: BedrockMessage;
  messageIndex: number;
}): void {
  if (!Array.isArray(message.content)) return;

  let toolCallIndex = 0;
  message.content.forEach((content) => {
    if (isToolUseContent(content)) {
      const toolCallPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}`;
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
 * Handles tool results within a message using OpenInference semantic conventions
 * Extracts tool result information and sets tool call ID references
 *
 * @param params Object containing tool result processing parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.message The Bedrock message containing tool results
 * @param params.messageIndex The message index in the conversation
 */
function handleToolResultsInMessage({
  span,
  message,
  messageIndex,
}: {
  span: Span;
  message: BedrockMessage;
  messageIndex: number;
}): void {
  if (!Array.isArray(message.content)) return;

  const toolResultBlocks = extractToolResultBlocks(message.content);
  toolResultBlocks.forEach((contentBlock) => {
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_TOOL_CALL_ID}`,
      contentBlock.tool_use_id,
    );
  });
}

/**
 * Adds detailed message content structure attributes for multi-modal content
 * Processes text, image, and other content types with appropriate OpenInference attributes
 *
 * @param params Object containing content processing parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.message The Bedrock message containing structured content
 * @param params.messageIndex The message index in the conversation
 */
function addMessageContentAttributes({
  span,
  message,
  messageIndex,
}: {
  span: Span;
  message: BedrockMessage;
  messageIndex: number;
}): void {
  if (!message.content) return;

  if (Array.isArray(message.content)) {
    message.content.forEach((content, contentIndex) => {
      const contentPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}`;

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
      } else if (isImageContent(content)) {
        setSpanAttribute(
          span,
          `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`,
          "image",
        );
        const imageUrl = formatImageUrl(content.source);
        setSpanAttribute(
          span,
          `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_IMAGE}.${SemanticConventions.IMAGE_URL}`,
          imageUrl,
        );
      }
    });
  }
}

/**
 * Extracts base request attributes for OpenInference semantic conventions
 * Sets fundamental LLM attributes including model, system, provider, and invocation parameters
 *
 * @param params Object containing extraction parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.command The InvokeModelCommand containing model ID
 * @param params.requestBody The parsed request body containing model parameters
 */
function extractBaseRequestAttributes({
  span,
  command,
  requestBody,
  system,
}: {
  span: Span;
  command: InvokeModelCommand;
  requestBody: InvokeModelRequestBody;
  system: LLMSystem;
}): void {
  const modelId = command.input?.modelId || "unknown";
  setSpanAttribute(
    span,
    SemanticConventions.LLM_MODEL_NAME,
    extractModelName(modelId),
  );

  const inputValue = JSON.stringify(requestBody);
  setSpanAttribute(span, SemanticConventions.INPUT_VALUE, inputValue);
  setSpanAttribute(span, SemanticConventions.INPUT_MIME_TYPE, MimeType.JSON);

  const invocationParams = extractInvocationParameters(requestBody, system);
  if (Object.keys(invocationParams).length > 0) {
    setSpanAttribute(
      span,
      SemanticConventions.LLM_INVOCATION_PARAMETERS,
      JSON.stringify(invocationParams),
    );
  }
}

/**
 * Extracts input messages attributes from InvokeModel request body
 * Processes message array and sets OpenInference input message attributes with proper indexing
 *
 * @param params Object containing extraction parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.requestBody The parsed request body containing messages array
 */
function extractInputMessagesAttributes({
  span,
  requestBody,
  system,
}: {
  span: Span;
  requestBody: InvokeModelRequestBody;
  system: LLMSystem;
}): void {
  const messages = normalizeRequestContentBlocks(requestBody, system);

  if (messages && Array.isArray(messages)) {
    messages.forEach((message, index) => {
      addMessageAttributes({ span, message, index });
    });
  }
}

/**
 * Extracts input tool attributes from InvokeModel request body
 * Processes tool definitions array and sets them as JSON schema attributes
 * Out of supported models, Anthropic, Amazon Nova, and Mistral Chat Completion
 * support tool calls.
 *
 * @param params Object containing extraction parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.requestBody The parsed request body containing tools array
 */
function extractInputToolAttributes({
  span,
  requestBody,
  system,
}: {
  span: Span;
  requestBody: InvokeModelRequestBody;
  system: LLMSystem;
}): void {
  if (
    system === LLMSystem.AMAZON &&
    requestBody.toolConfig &&
    isObjectWithStringKeys(requestBody.toolConfig) &&
    requestBody.toolConfig.tools &&
    Array.isArray(requestBody.toolConfig.tools)
  ) {
    requestBody.toolConfig.tools.forEach((tool, index) => {
      if (tool.toolSpec && isObjectWithStringKeys(tool.toolSpec)) {
        setSpanAttribute(
          span,
          `${SemanticConventions.LLM_TOOLS}.${index}.${SemanticConventions.TOOL_JSON_SCHEMA}`,
          JSON.stringify(tool.toolSpec),
        );
      }
    });
  } else if (requestBody.tools && Array.isArray(requestBody.tools)) {
    requestBody.tools.forEach((tool, index) => {
      // Both Anthropic and Mistral Chat Completion have the tool definitions in the same place
      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_TOOLS}.${index}.${SemanticConventions.TOOL_JSON_SCHEMA}`,
        JSON.stringify(tool),
      );
    });
  }
}

/**
 * Extracts semantic convention attributes from InvokeModel request command
 * Main entry point for processing InvokeModel API requests with comprehensive error handling
 *
 * Processes:
 * - Request body parsing from multiple formats
 * - Base model and system attributes
 * - Input messages with multi-modal content
 * - Tool definitions
 *
 * @param params Object containing extraction parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.command The InvokeModelCommand to extract attributes from
 */
export const extractInvokeModelRequestAttributes = withSafety({
  fn: ({
    span,
    command,
    system,
  }: {
    span: Span;
    command: InvokeModelCommand;
    system: LLMSystem;
  }): void => {
    const requestBody = parseRequestBody(command);
    if (!requestBody) {
      return;
    }

    extractBaseRequestAttributes({ span, command, requestBody, system });
    extractInputMessagesAttributes({ span, requestBody, system });
    extractInputToolAttributes({ span, requestBody, system });
  },
  onError: (error) => {
    diag.warn("Error extracting InvokeModel request attributes:", error);
  },
});
