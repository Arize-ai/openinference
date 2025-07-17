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
import { InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import { InvokeModelRequestBody, BedrockMessage } from "../types/bedrock-types";
import {
  extractTextFromContent,
  extractToolResultBlocks,
  formatImageUrl,
} from "../utils/content-processing";

/**
 * Extracts base request attributes (model, system, provider, invocation parameters)
 */
export function extractBaseRequestAttributes(
  command: InvokeModelCommand,
): Record<string, any> {
  const modelId = command.input?.modelId || "unknown";
  const requestBody = parseRequestBody(command);

  const attributes: Record<string, any> = {
    [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
    [SemanticConventions.LLM_SYSTEM]: getSystemFromModelId(modelId),
    [SemanticConventions.LLM_MODEL_NAME]: extractModelName(modelId),
    [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
    [SemanticConventions.LLM_PROVIDER]: LLMProvider.AWS,
  };

  // Add invocation parameters for model configuration
  const invocationParams = extractInvocationParameters(requestBody);
  if (Object.keys(invocationParams).length > 0) {
    attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS] =
      JSON.stringify(invocationParams);
  }

  return attributes;
}

/**
 * Extracts input messages attributes from request body
 */
export function extractInputMessagesAttributes(
  requestBody: InvokeModelRequestBody,
): Record<string, any> {
  const attributes: Record<string, any> = {};

  // Extract user's message text as primary input value
  const inputValue = extractPrimaryInputValue(requestBody);
  attributes[SemanticConventions.INPUT_VALUE] = inputValue;

  // Add structured input message attributes
  if (requestBody.messages && Array.isArray(requestBody.messages)) {
    requestBody.messages.forEach((message, index) => {
      addMessageAttributes(attributes, message, index);
    });
  }

  return attributes;
}

/**
 * Extracts input tool attributes from request body
 */
export function extractInputToolAttributes(
  requestBody: InvokeModelRequestBody,
): Record<string, any> {
  const attributes: Record<string, any> = {};

  // Add input tools from request if present
  if (requestBody.tools && Array.isArray(requestBody.tools)) {
    requestBody.tools.forEach((tool, index) => {
      // Convert Bedrock tool format to OpenAI format for consistent schema
      const openAIToolFormat = {
        type: "function",
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.input_schema,
        },
      };
      attributes[
        `${SemanticConventions.LLM_TOOLS}.${index}.${SemanticConventions.TOOL_JSON_SCHEMA}`
      ] = JSON.stringify(openAIToolFormat);
    });
  }

  return attributes;
}

/**
 * Extracts semantic convention attributes from InvokeModel request command
 */
export function extractInvokeModelRequestAttributes(
  command: InvokeModelCommand,
): Record<string, any> {
  const requestBody = parseRequestBody(command);

  // Start with base attributes
  const attributes = extractBaseRequestAttributes(command);

  // Add input messages attributes
  Object.assign(attributes, extractInputMessagesAttributes(requestBody));

  // Add input tool attributes
  Object.assign(attributes, extractInputToolAttributes(requestBody));

  return attributes;
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
      bodyString = new TextDecoder().decode(command.input.body);
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
 * Extracts invocation parameters from request body
 */
function extractInvocationParameters(
  requestBody: InvokeModelRequestBody,
): Record<string, any> {
  const invocationParams: Record<string, any> = {};

  if (requestBody.anthropic_version) {
    invocationParams.anthropic_version = requestBody.anthropic_version;
  }
  if (requestBody.max_tokens) {
    invocationParams.max_tokens = requestBody.max_tokens;
  }
  if (requestBody.temperature !== undefined) {
    invocationParams.temperature = requestBody.temperature;
  }
  if (requestBody.top_p !== undefined) {
    invocationParams.top_p = requestBody.top_p;
  }
  if (requestBody.top_k !== undefined) {
    invocationParams.top_k = requestBody.top_k;
  }

  return invocationParams;
}

/**
 * Extracts the primary input value as full request body JSON
 */
function extractPrimaryInputValue(requestBody: InvokeModelRequestBody): string {
  // Following OpenAI and LangChain pattern: use full request body as JSON
  return JSON.stringify(requestBody);
}

/**
 * Adds message attributes to the attributes object
 */
function addMessageAttributes(
  attributes: Record<string, any>,
  message: BedrockMessage,
  index: number,
): void {
  if (!message.role) return;

  attributes[
    `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_ROLE}`
  ] = message.role;

  // Determine if this is a simple text message or complex multimodal message
  const isSimpleTextMessage = typeof message.content === "string" || 
    (Array.isArray(message.content) && message.content.length === 1 && message.content[0].type === "text");

  if (isSimpleTextMessage) {
    // For simple text messages, use only the top-level message.content
    const messageContent = extractTextFromContent(message.content);
    if (messageContent) {
      attributes[
        `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_CONTENT}`
      ] = messageContent;
    }
  } else {
    // For complex multimodal messages, use only the detailed message.contents structure
    addMessageContentAttributes(attributes, message, index);
  }

  // Handle complex content arrays (tool results, etc.)
  if (Array.isArray(message.content)) {
    const toolResultBlocks = extractToolResultBlocks(message.content);
    toolResultBlocks.forEach((contentBlock, contentIndex: number) => {
      // Extract tool result attributes
      attributes[
        `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${contentIndex}.${SemanticConventions.TOOL_CALL_ID}`
      ] = contentBlock.tool_use_id;
      attributes[
        `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${contentIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
      ] = JSON.stringify({ result: contentBlock.content });
    });
  }
}

/**
 * Adds detailed message content structure attributes
 * Phase 2: Message Content Structure Enhancement
 */
function addMessageContentAttributes(
  attributes: Record<string, any>,
  message: BedrockMessage,
  messageIndex: number,
): void {
  if (!message.content) return;

  // Handle string content
  if (typeof message.content === "string") {
    attributes[
      `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_TYPE}`
    ] = "text";
    attributes[
      `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_TEXT}`
    ] = message.content;
    return;
  }

  // Handle array content (multi-modal)
  if (Array.isArray(message.content)) {
    message.content.forEach((content, contentIndex) => {
      const contentPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}`;
      
      if (content.type === "text") {
        attributes[`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "text";
        attributes[`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = content.text;
      } else if (content.type === "image") {
        attributes[`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "image";
        if (content.source) {
          const imageUrl = formatImageUrl(content.source);
          attributes[`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_IMAGE}.${SemanticConventions.IMAGE_URL}`] = imageUrl;
        }
      } else if (content.type === "tool_use") {
        attributes[`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "tool_use";
        attributes[`${contentPrefix}.message_content.tool_use.name`] = content.name;
        attributes[`${contentPrefix}.message_content.tool_use.input`] = JSON.stringify(content.input);
        attributes[`${contentPrefix}.message_content.tool_use.id`] = content.id;
      } else if (content.type === "tool_result") {
        attributes[`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "tool_result";
        attributes[`${contentPrefix}.message_content.tool_result.tool_use_id`] = content.tool_use_id;
        attributes[`${contentPrefix}.message_content.tool_result.content`] = content.content;
      }
    });
  }
}
