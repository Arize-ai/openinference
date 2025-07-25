import { Span, AttributeValue } from "@opentelemetry/api";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";
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
 * Aggregates multiple system prompts into a single string
 * Concatenates all text content from system prompts with double newline separation
 * 
 * @param systemPrompts Array of system content blocks from Bedrock Converse API
 * @returns {string} Combined system prompt text with proper formatting
 */
export function aggregateSystemPrompts(systemPrompts: SystemContentBlock[]): string {
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
 * Extracts OpenInference semantic convention attributes from a single Bedrock message
 * Handles role, content, tool calls, and tool results following the OpenInference specification
 * 
 * @param message The Bedrock message to extract attributes from
 * @returns {Record<string, AttributeValue>} Object containing semantic convention attributes
 */
export function getAttributesFromMessage(
  message: Message,
): Record<string, AttributeValue> {
  const attributes: Record<string, AttributeValue> = {};

  if (message.role) {
    attributes[SemanticConventions.MESSAGE_ROLE] = message.role;
  }

  if (message.content) {
    let toolCallIndex = 0;

    for (const [index, content] of message.content.entries()) {
      // Process content as our custom types for attribute extraction
      const contentAttributes = getAttributesFromMessageContent(content);
      for (const [key, value] of Object.entries(contentAttributes)) {
        attributes[`${SemanticConventions.MESSAGE_CONTENTS}.${index}.${key}`] =
          value;
      }

      // Handle tool calls at the message level using proper semantic conventions
      if (isConverseToolUseContent(content)) {
        const toolCallPrefix = `${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}`;
        attributes[`${toolCallPrefix}.${SemanticConventions.TOOL_CALL_ID}`] =
          content.toolUse.toolUseId;
        attributes[
          `${toolCallPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
        ] = content.toolUse.name;
        if (content.toolUse.input) {
          attributes[
            `${toolCallPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
          ] = JSON.stringify(content.toolUse.input);
        }
        toolCallIndex++;
      } else if (isConverseToolResultContent(content)) {
        attributes[SemanticConventions.MESSAGE_TOOL_CALL_ID] =
          content.toolResult.toolUseId;
      }
    }
  }

  return attributes;
}

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
): Record<string, AttributeValue> {
  const attributes: Record<string, AttributeValue> = {};

  if (isConverseTextContent(content)) {
    attributes[SemanticConventions.MESSAGE_CONTENT_TYPE] = "text";
    attributes[SemanticConventions.MESSAGE_CONTENT_TEXT] = content.text;
  } else if (isConverseImageContent(content)) {
    attributes[SemanticConventions.MESSAGE_CONTENT_TYPE] = "image";
    if (content.image.source.bytes) {
      // Convert bytes to base64 data URL for consistent representation
      const base64 = Buffer.from(content.image.source.bytes).toString("base64");
      const mimeType = `image/${content.image.format}`;
      attributes[
        `${SemanticConventions.MESSAGE_CONTENT_IMAGE}.${SemanticConventions.IMAGE_URL}`
      ] = `data:${mimeType};base64,${base64}`;
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
  baseKey: typeof SemanticConventions.LLM_INPUT_MESSAGES | typeof SemanticConventions.LLM_OUTPUT_MESSAGES;
}): void {
  for (const [index, message] of messages.entries()) {
    const messageAttributes = getAttributesFromMessage(message);
    for (const [key, value] of Object.entries(messageAttributes)) {
      setSpanAttribute(span, `${baseKey}.${index}.${key}`, value);
    }
  }
}
