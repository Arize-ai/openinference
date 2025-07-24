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
 * Provides null-safe attribute setting for OpenTelemetry spans
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
 * Concatenates all text content from system prompts with proper formatting
 */
export function aggregateSystemPrompts(systemPrompts: SystemContentBlock[]): string {
  return systemPrompts
    .map((prompt) => prompt.text || "")
    .filter(Boolean)
    .join("\n\n");
}

/**
 * Aggregates system prompts with messages into a unified message array
 * System prompts are converted to a single system message at the beginning
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
 * Extracts attributes from a single message
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
 * Extracts attributes from message content
 * Note: We still use our custom content type guards here since AWS SDK ContentBlock
 * has complex union types that don't match our processing needs exactly
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
 * Processes multiple messages and sets attributes on span with proper indexing
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
