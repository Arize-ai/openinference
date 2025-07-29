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
import {
  InvokeModelCommand,
  InferenceConfiguration,
} from "@aws-sdk/client-bedrock-runtime";
import { withSafety } from "@arizeai/openinference-core";
import {
  InvokeModelRequestBody,
  BedrockMessage,
  isTextContent,
  isImageContent,
  isToolUseContent,
} from "../types/bedrock-types";
import {
  extractTextFromContent,
  extractToolResultBlocks,
  formatImageUrl,
} from "../utils/content-processing";
import { setSpanAttribute, extractModelName } from "./attribute-helpers";

/**
 * Safely parses the InvokeModel request body with comprehensive error handling
 * Handles multiple body formats (string, Buffer, Uint8Array, ArrayBuffer) and provides fallback
 *
 * @param command The InvokeModelCommand containing the request body to parse
 * @returns {InvokeModelRequestBody} Parsed request body or fallback structure on error
 */
const parseRequestBody = withSafety({
  fn: (command: InvokeModelCommand): InvokeModelRequestBody => {
    if (!command.input?.body) {
      throw new Error("Request body is missing");
    }

    let bodyString: string;
    if (typeof command.input.body === "string") {
      bodyString = command.input.body;
    } else if (Buffer.isBuffer(command.input.body)) {
      bodyString = command.input.body.toString("utf8");
    } else if (command.input.body instanceof Uint8Array) {
      bodyString = new TextDecoder().decode(command.input.body);
    } else if (command.input.body instanceof ArrayBuffer) {
      bodyString = new TextDecoder().decode(new Uint8Array(command.input.body));
    } else {
      // For other types, convert to string safely
      bodyString = String(command.input.body);
    }
    return JSON.parse(bodyString) as InvokeModelRequestBody;
  },
  onError: (error) => {
    diag.warn("Error parsing InvokeModel request body:", error);
    return null;
  },
});

/**
 * Extracts invocation parameters from request body using AWS SDK standards
 * Maps snake_case parameter names to camelCase AWS SDK convention where applicable
 * Combines standard AWS SDK InferenceConfiguration with vendor-specific parameters
 *
 * @param requestBody The parsed request body containing model parameters
 * @returns {ExtractedInvocationParameters} Object containing extracted parameters
 */
function extractInvocationParameters(
  requestBody: InvokeModelRequestBody,
  system: LLMSystem,
): Record<string, unknown> {
  if (system === LLMSystem.AMAZON && requestBody.inferenceConfig && 
      typeof requestBody.inferenceConfig === 'object' && requestBody.inferenceConfig !== null) {
    return requestBody.inferenceConfig as Record<string, unknown>;
  } else if (system === LLMSystem.AMAZON && requestBody.textGenerationConfig && 
             typeof requestBody.textGenerationConfig === 'object' && requestBody.textGenerationConfig !== null) {
    return requestBody.textGenerationConfig as Record<string, unknown>;
  } else {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const {system, messages, tools, prompt, ...invocationParams} = requestBody;
    return invocationParams;
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
  const inputValue = JSON.stringify(requestBody);
  setSpanAttribute(span, SemanticConventions.INPUT_VALUE, inputValue);

  if (requestBody.messages && Array.isArray(requestBody.messages)) {
    requestBody.messages.forEach((message, index) => {
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
  if (requestBody.tools && Array.isArray(requestBody.tools)) {
    requestBody.tools.forEach((tool, index) => {
      // Amazon Nova has one extra nest for it's tool declarations
      if (system === LLMSystem.AMAZON && tool.toolSpec) {
        setSpanAttribute(
          span,
          `${SemanticConventions.LLM_TOOLS}.${index}.${SemanticConventions.TOOL_JSON_SCHEMA}`,
          JSON.stringify(tool.toolSpec),
        );
      }
      // Both Anthropic and Mistral Chat Completion have the tool definitions in the same place
      else {
        setSpanAttribute(
          span,
          `${SemanticConventions.LLM_TOOLS}.${index}.${SemanticConventions.TOOL_JSON_SCHEMA}`,
          JSON.stringify(tool),
        );
      } 
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

  const isSimpleTextMessage =
    typeof message.content === "string" ||
    (Array.isArray(message.content) &&
      message.content.length === 1 &&
      isTextContent(message.content[0]));

  if (isSimpleTextMessage) {
    const messageContent = extractTextFromContent(message.content);
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_CONTENT}`,
      messageContent,
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
