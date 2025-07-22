/**
 * Request attribute extraction for AWS Bedrock instrumentation
 *
 * Handles extraction of semantic convention attributes from InvokeModel requests including:
 * - Base model and system attributes
 * - Input message processing
 * - Tool definition processing
 * - Invocation parameters
 */

import {
  SemanticConventions,
  OpenInferenceSpanKind,
  MimeType,
  LLMProvider,
} from "@arizeai/openinference-semantic-conventions";
import { InvokeModelCommand, InferenceConfiguration } from "@aws-sdk/client-bedrock-runtime";
import { InvokeModelRequestBody, BedrockMessage } from "../types/bedrock-types";
import {
  extractTextFromContent,
  extractToolResultBlocks,
  formatImageUrl,
} from "../utils/content-processing";
import { setSpanAttribute } from "./attribute-helpers";
import { Span } from "@opentelemetry/api";

/**
 * Extracts base request attributes (model, system, provider, invocation parameters)
 */
export function extractBaseRequestAttributes({
  span,
  command,
}: {
  span: Span;
  command: InvokeModelCommand;
}): void {
  const modelId = command.input?.modelId || "unknown";
  const requestBody = parseRequestBody(command);

  // Set base attributes individually with null checking
  setSpanAttribute(
    span,
    SemanticConventions.OPENINFERENCE_SPAN_KIND,
    OpenInferenceSpanKind.LLM,
  );
  setSpanAttribute(
    span,
    SemanticConventions.LLM_SYSTEM,
    getSystemFromModelId(modelId),
  );
  setSpanAttribute(
    span,
    SemanticConventions.LLM_MODEL_NAME,
    extractModelName(modelId),
  );
  setSpanAttribute(span, SemanticConventions.INPUT_MIME_TYPE, MimeType.JSON);
  setSpanAttribute(span, SemanticConventions.LLM_PROVIDER, LLMProvider.AWS);

  // Add invocation parameters for model configuration
  const invocationParams = extractInvocationParameters(requestBody);
  if (Object.keys(invocationParams).length > 0) {
    setSpanAttribute(
      span,
      SemanticConventions.LLM_INVOCATION_PARAMETERS,
      JSON.stringify(invocationParams),
    );
  }
}

/**
 * Extracts input messages attributes from request body
 */
export function extractInputMessagesAttributes({
  span,
  requestBody,
}: {
  span: Span;
  requestBody: InvokeModelRequestBody;
}): void {
  // Extract user's message text as primary input value
  const inputValue = extractPrimaryInputValue(requestBody);
  setSpanAttribute(span, SemanticConventions.INPUT_VALUE, inputValue);

  // Add structured input message attributes
  if (requestBody.messages && Array.isArray(requestBody.messages)) {
    requestBody.messages.forEach((message, index) => {
      addMessageAttributes({ span, message, index });
    });
  }
}

/**
 * Extracts input tool attributes from request body
 */
export function extractInputToolAttributes({
  span,
  requestBody,
}: {
  span: Span;
  requestBody: InvokeModelRequestBody;
}): void {
  // Add input tools from request if present
  if (requestBody.tools && Array.isArray(requestBody.tools)) {
    requestBody.tools.forEach((tool, index) => {
      // Convert Bedrock tool format to function schema for consistent format
      const toolFormat = {
        type: "function",
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.input_schema,
        },
      };
      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_TOOLS}.${index}.${SemanticConventions.TOOL_JSON_SCHEMA}`,
        JSON.stringify(toolFormat),
      );
    });
  }
}

/**
 * Extracts semantic convention attributes from InvokeModel request command
 */
export function extractInvokeModelRequestAttributes({
  span,
  command,
}: {
  span: Span;
  command: InvokeModelCommand;
}): void {
  const requestBody = parseRequestBody(command);

  // Extract base attributes
  extractBaseRequestAttributes({ span, command });

  // Add input messages attributes
  extractInputMessagesAttributes({ span, requestBody });

  // Add input tool attributes
  extractInputToolAttributes({ span, requestBody });
}

// Helper functions

/**
 * Extracts vendor-specific system name from model ID
 */
function getSystemFromModelId(modelId: string): string {
  if (modelId.includes("anthropic")) return "anthropic";
  if (modelId.includes("ai21")) return "ai21";
  if (modelId.includes("amazon")) return "amazon";
  if (modelId.includes("cohere")) return "cohere";
  if (modelId.includes("meta")) return "meta";
  if (modelId.includes("mistral")) return "mistral";
  return "bedrock"; // fallback
}

/**
 * Extracts clean model name from full model ID
 */
function extractModelName(modelId: string): string {
  // "anthropic.claude-3-haiku-20240307-v1:0" -> "claude-3-haiku-20240307"
  // "amazon.titan-text-express-v1" -> "titan-text-express-v1"
  const parts = modelId.split(".");
  if (parts.length > 1) {
    const modelPart = parts[1];
    // For Anthropic models, remove version suffix like "-v1:0"
    if (modelId.includes("anthropic")) {
      const versionIndex = modelPart.indexOf("-v");
      if (versionIndex > 0) {
        return modelPart.substring(0, versionIndex);
      }
    }
    return modelPart;
  }
  return modelId;
}

/**
 * Safely parses the request body JSON
 */
function parseRequestBody(command: InvokeModelCommand): InvokeModelRequestBody {
  try {
    if (!command.input?.body) {
      return {} as InvokeModelRequestBody;
    }

    // Handle both string (test format) and Uint8Array (SDK format)
    let bodyString: string;
    if (typeof command.input.body === "string") {
      bodyString = command.input.body;
    } else if (command.input.body instanceof Uint8Array) {
      bodyString = new TextDecoder().decode(command.input.body as Uint8Array);
    } else {
      // Handle other blob types by converting to Uint8Array first
      bodyString = new TextDecoder().decode(
        new Uint8Array(command.input.body as ArrayBuffer),
      );
    }

    return JSON.parse(bodyString);
  } catch (error) {
    return {} as InvokeModelRequestBody;
  }
}

/**
 * Interface for invocation parameters combining AWS SDK standard interface with vendor-specific parameters
 * Uses AWS SDK's InferenceConfiguration for standard parameters across all model vendors
 */
interface ExtractedInvocationParameters extends Partial<InferenceConfiguration> {
  // Vendor-specific parameters currently supported in the existing codebase
  top_k?: number;                    // Used by Anthropic and other models
  anthropic_version?: string;        // Anthropic-specific API version
}

/**
 * Extracts invocation parameters from request body using AWS SDK standards
 * Maps snake_case parameter names to camelCase AWS SDK convention where applicable
 */
function extractInvocationParameters(
  requestBody: InvokeModelRequestBody,
): ExtractedInvocationParameters {
  const invocationParams: ExtractedInvocationParameters = {};

  // Standard AWS SDK parameters (using camelCase naming)
  if (requestBody.max_tokens) {
    invocationParams.maxTokens = requestBody.max_tokens;
  }
  if (requestBody.temperature !== undefined) {
    invocationParams.temperature = requestBody.temperature;
  }
  if (requestBody.top_p !== undefined) {
    invocationParams.topP = requestBody.top_p;
  }
  if (requestBody.stop_sequences) {
    invocationParams.stopSequences = requestBody.stop_sequences;
  }

  // Vendor-specific parameters currently supported in the existing codebase
  if (requestBody.top_k !== undefined) {
    invocationParams.top_k = requestBody.top_k;
  }
  if (requestBody.anthropic_version) {
    invocationParams.anthropic_version = requestBody.anthropic_version;
  }

  return invocationParams;
}

/**
 * Extracts the primary input value as full request body JSON
 */
function extractPrimaryInputValue(requestBody: InvokeModelRequestBody): string {
  // Use full request body as JSON
  return JSON.stringify(requestBody);
}

/**
 * Adds message attributes to the span
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

  // Determine if this is a simple text message or complex multimodal message
  const isSimpleTextMessage =
    typeof message.content === "string" ||
    (Array.isArray(message.content) &&
      message.content.length === 1 &&
      message.content[0].type === "text");

  if (isSimpleTextMessage) {
    // For simple text messages, use only the top-level message.content
    const messageContent = extractTextFromContent(message.content);
    if (messageContent) {
      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_CONTENT}`,
        messageContent,
      );
    }
      } else {
      // For complex multimodal messages, use only the detailed message.contents structure
      addMessageContentAttributes({ span, message, messageIndex: index });
    }

  // Handle complex content arrays (tool results, etc.)
  if (Array.isArray(message.content)) {
    const toolResultBlocks = extractToolResultBlocks(message.content);
    toolResultBlocks.forEach((contentBlock, contentIndex: number) => {
      // Extract tool result attributes
      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${contentIndex}.${SemanticConventions.TOOL_CALL_ID}`,
        contentBlock.tool_use_id,
      );
      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${contentIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`,
        JSON.stringify({ result: contentBlock.content }),
      );
    });
  }
}

/**
 * Adds detailed message content structure attributes
 * Message Content Structure Enhancement
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

  // Handle string content
  if (typeof message.content === "string") {
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_TYPE}`,
      "text",
    );
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_TEXT}`,
      message.content,
    );
    return;
  }

  // Handle array content (multi-modal)
  if (Array.isArray(message.content)) {
    message.content.forEach((content, contentIndex) => {
      const contentPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}`;

      if (content.type === "text") {
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
      } else if (content.type === "image") {
        setSpanAttribute(
          span,
          `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`,
          "image",
        );
        if (content.source) {
          const imageUrl = formatImageUrl(content.source);
          setSpanAttribute(
            span,
            `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_IMAGE}.${SemanticConventions.IMAGE_URL}`,
            imageUrl,
          );
        }
      } else if (content.type === "tool_use") {
        setSpanAttribute(
          span,
          `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`,
          "tool_use",
        );
        setSpanAttribute(
          span,
          `${contentPrefix}.message_content.tool_use.name`,
          content.name,
        );
        setSpanAttribute(
          span,
          `${contentPrefix}.message_content.tool_use.input`,
          JSON.stringify(content.input),
        );
        setSpanAttribute(
          span,
          `${contentPrefix}.message_content.tool_use.id`,
          content.id,
        );
      } else if (content.type === "tool_result") {
        setSpanAttribute(
          span,
          `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`,
          "tool_result",
        );
        setSpanAttribute(
          span,
          `${contentPrefix}.message_content.tool_result.tool_use_id`,
          content.tool_use_id,
        );
        setSpanAttribute(
          span,
          `${contentPrefix}.message_content.tool_result.content`,
          content.content,
        );
      }
    });
  }
}
