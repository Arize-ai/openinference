import { Span, AttributeValue, Attributes, diag } from "@opentelemetry/api";
import { withSafety, safelyJSONStringify } from "@arizeai/openinference-core";
import {
  SemanticConventions,
  LLMSystem,
  OpenInferenceSpanKind,
  LLMProvider,
} from "@arizeai/openinference-semantic-conventions";
import {
  Message,
  SystemContentBlock,
  ContentBlock,
  ConversationRole,
} from "@aws-sdk/client-bedrock-runtime";
import {
  isConverseTextContent,
  isConverseImageContent,
  isConverseToolUseContent,
  isConverseToolResultContent,
} from "../types/bedrock-types";
import { formatImageUrl } from "./invoke-model-helpers";

/**
 * Sets a span attribute only if the value is not null, undefined, or empty string
 * Provides null-safe attribute setting for OpenTelemetry spans to avoid polluting traces with empty values
 *
 * @param span The OpenTelemetry span to set the attribute on
 * @param key The attribute key following OpenInference semantic conventions
 * @param value The attribute value to set, will be skipped if null/undefined/empty
 */
export function setSpanAttribute(
  span: Span,
  key: string,
  value: AttributeValue | null | undefined,
): void {
  if (value != null && value !== "") {
    span.setAttribute(key, value);
  }
}

/**
 * Extracts vendor-specific system name from Bedrock model ID
 * Maps model IDs to their corresponding AI system providers
 *
 * @param modelId The full Bedrock model identifier (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
 * @returns {string} The system provider name (e.g., "anthropic", "meta", "mistral") or "bedrock" as fallback
 */
export function getSystemFromModelId(modelId: string): LLMSystem {
  if (modelId.includes("anthropic")) return LLMSystem.ANTHROPIC;
  if (modelId.includes("ai21")) return LLMSystem.AI21;
  if (modelId.includes("amazon")) return LLMSystem.AMAZON;
  if (modelId.includes("cohere")) return LLMSystem.COHERE;
  if (modelId.includes("meta")) return LLMSystem.META;
  if (modelId.includes("mistral")) return LLMSystem.MISTRALAI;
  return LLMSystem.AMAZON;
}

export function setBasicSpanAttributes(span: Span, llm_system: LLMSystem) {
  setSpanAttribute(span, SemanticConventions.LLM_PROVIDER, LLMProvider.AWS);

  setSpanAttribute(
    span,
    SemanticConventions.OPENINFERENCE_SPAN_KIND,
    OpenInferenceSpanKind.LLM,
  );

  setSpanAttribute(span, SemanticConventions.LLM_SYSTEM, llm_system);
}

/**
 * Aggregates multiple system prompts into a single string
 * Concatenates all text content from system prompts with double newline separation
 *
 * @param systemPrompts Array of system content blocks from Bedrock Converse API
 * @returns {string} Combined system prompt text with proper formatting
 */
export function aggregateSystemPrompts(
  systemPrompts: SystemContentBlock[],
): string {
  return systemPrompts
    .map((prompt) => prompt.text || "")
    .filter(Boolean)
    .join("\n\n");
}

/**
 * Aggregates system prompts with messages into a unified message array
 * System prompts are converted to a single system message at the beginning of the conversation
 *
 * @param systemPrompts Array of system content blocks to convert to system message
 * @param messages Array of conversation messages
 * @returns {Message[]} Combined message array with system prompt as first message if present
 */
export function aggregateMessages(
  systemPrompts: SystemContentBlock[] = [],
  messages: Message[] = [],
): Message[] {
  const aggregated: Message[] = [];

  if (systemPrompts.length > 0) {
    aggregated.push({
      role: "system" as ConversationRole,
      content: [{ text: aggregateSystemPrompts(systemPrompts) }],
    });
  }

  return [...aggregated, ...messages];
}

/**
 * Safely converts supported byte-like inputs to base64.
 * Handles: Uint8Array, Buffer.
 * On unsupported inputs, logs a warning and returns undefined.
 */
const toBase64ImageBytes = withSafety({
  fn: (bytes: Uint8Array | Buffer): string | undefined => {
    if (Buffer.isBuffer(bytes)) {
      return (bytes as Buffer).toString("base64");
    }
    if (bytes instanceof Uint8Array) {
      return Buffer.from(bytes).toString("base64");
    }
    diag.warn("Unsupported image bytes type encountered");
    return undefined;
  },
  onError: (error) => {
    diag.warn("Failed to convert image bytes to base64", error as Error);
  },
});

/**
 * Extracts OpenInference semantic convention attributes from message content blocks
 * Handles text, image, and tool content types with appropriate attribute mapping
 *
 * Note: Uses custom content type guards since AWS SDK ContentBlock union types
 * require additional processing for reliable type detection
 *
 * @param content The content block to extract attributes from
 * @returns {Record<string, AttributeValue>} Object containing content-specific attributes
 */
export function getAttributesFromMessageContent(
  content: ContentBlock,
): Attributes {
  const attributes: Attributes = {};

  if (isConverseTextContent(content)) {
    attributes[SemanticConventions.MESSAGE_CONTENT_TYPE] = "text";
    attributes[SemanticConventions.MESSAGE_CONTENT_TEXT] = content.text;
  } else if (isConverseImageContent(content)) {
    attributes[SemanticConventions.MESSAGE_CONTENT_TYPE] = "image";
    const base64 = toBase64ImageBytes(content.image.source.bytes);
    if (base64) {
      const mimeType = `image/${content.image.format}`;
      attributes[
        `${SemanticConventions.MESSAGE_CONTENT_IMAGE}.${SemanticConventions.IMAGE_URL}`
      ] = formatImageUrl({
        type: "base64",
        data: base64,
        media_type: mimeType,
      });
    }
    // Add format attribute for image content
    attributes[`${SemanticConventions.MESSAGE_CONTENT_IMAGE}.format`] =
      content.image.format;
  }

  return attributes;
}

/**
 * Extracts OpenInference semantic convention attributes from a single Bedrock message
 * Handles role, content, tool calls, and tool results following the OpenInference specification
 *
 * @param message The Bedrock message to extract attributes from
 * @returns {Record<string, AttributeValue>} Object containing semantic convention attributes
 */
export function getAttributesFromMessage(message: Message): Attributes {
  const attributes: Attributes = {};

  if (message.role) {
    attributes[SemanticConventions.MESSAGE_ROLE] = message.role;
  }

  if (message.content) {
    // Check if this is a simple single text content case
    if (
      message.content.length === 1 &&
      isConverseTextContent(message.content[0])
    ) {
      // Use simple format for single text content
      attributes[SemanticConventions.MESSAGE_CONTENT] = message.content[0].text;
    } else {
      // Use complex format for multi-modal or multi-content messages
      let toolCallIndex = 0;

      for (const [index, content] of message.content.entries()) {
        // Process content as our custom types for attribute extraction
        const contentAttributes = getAttributesFromMessageContent(content);
        for (const key in contentAttributes) {
          attributes[
            `${SemanticConventions.MESSAGE_CONTENTS}.${index}.${key}`
          ] = contentAttributes[key];
        }

        // Handle tool calls at the message level using proper semantic conventions
        if (isConverseToolUseContent(content)) {
          const toolCallPrefix = `${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}`;
          attributes[`${toolCallPrefix}.${SemanticConventions.TOOL_CALL_ID}`] =
            content.toolUse.toolUseId;
          attributes[
            `${toolCallPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
          ] = content.toolUse.name;
          const argsJson = safelyJSONStringify(content.toolUse.input);
          if (argsJson) {
            attributes[
              `${toolCallPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
            ] = argsJson;
          }
          toolCallIndex++;
        } else if (isConverseToolResultContent(content)) {
          attributes[SemanticConventions.MESSAGE_TOOL_CALL_ID] =
            content.toolResult.toolUseId;
        }
      }
    }
  }

  return attributes;
}

/**
 * Processes multiple messages and sets OpenInference attributes on span with proper indexing
 * Iterates through messages array and applies semantic convention attributes with message index
 *
 * @param params Object containing processing parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.messages Array of messages to process
 * @param params.baseKey Base semantic convention key (either input or output messages)
 */
export function processMessages({
  span,
  messages,
  baseKey,
}: {
  span: Span;
  messages: Message[];
  baseKey:
    | typeof SemanticConventions.LLM_INPUT_MESSAGES
    | typeof SemanticConventions.LLM_OUTPUT_MESSAGES;
}): void {
  for (const [index, message] of messages.entries()) {
    const messageAttributes = getAttributesFromMessage(message);
    for (const [key, value] of Object.entries(messageAttributes)) {
      setSpanAttribute(span, `${baseKey}.${index}.${key}`, value);
    }
  }
}

/**
 * Extracts clean model name from full Bedrock model ID
 * Removes vendor prefix and version suffixes to get the base model name
 *
 * @param modelId The full Bedrock model identifier
 * @returns {string} The cleaned model name (e.g., "claude-3-sonnet" from "anthropic.claude-3-sonnet-20240229-v1:0")
 */
export function extractModelName(modelId: string): string {
  const parts = modelId.split(".");
  if (parts.length > 1) {
    const modelPart = parts[1];
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
