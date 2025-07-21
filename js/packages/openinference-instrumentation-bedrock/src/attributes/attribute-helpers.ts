import { Span, AttributeValue } from "@opentelemetry/api";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";
import {
  SystemPrompt,
  ConverseMessage,
  ConverseContentBlock,
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
 * Sets multiple span attributes with null checking
 */
export function setSpanAttributes(
  span: Span,
  attributes: Record<string, AttributeValue | null | undefined>,
) {
  Object.entries(attributes).forEach(([key, value]) => {
    setSpanAttribute(span, key, value);
  });
}

/**
 * Aggregates multiple system prompts into a single string with space separation
 */
export function aggregateSystemPrompts(systemPrompts: SystemPrompt[]): string {
  return systemPrompts
    .map((prompt) => prompt.text)
    .join(" ")
    .trim();
}

/**
 * Aggregates system prompts and messages into a single message array
 */
export function aggregateMessages(
  systemPrompts: SystemPrompt[] = [],
  messages: ConverseMessage[] = [],
): ConverseMessage[] {
  const aggregated: ConverseMessage[] = [];

  if (systemPrompts.length > 0) {
    aggregated.push({
      role: "system" as const,
      content: [{ text: aggregateSystemPrompts(systemPrompts) }],
    });
  }

  aggregated.push(...messages);
  return aggregated;
}

/**
 * Extracts attributes from a single message
 */
export function getAttributesFromMessage(
  message: ConverseMessage,
): Record<string, AttributeValue> {
  const attributes: Record<string, AttributeValue> = {};

  if (message.role) {
    attributes[SemanticConventions.MESSAGE_ROLE] = message.role;
  }

  if (message.content) {
    let toolCallIndex = 0;

    for (const [index, content] of message.content.entries()) {
      const contentAttributes = getAttributesFromMessageContent(content);
      for (const [key, value] of Object.entries(contentAttributes)) {
        attributes[`${SemanticConventions.MESSAGE_CONTENTS}.${index}.${key}`] =
          value;
      }

      // Handle tool calls and tool results using proper semantic conventions
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
 */
export function getAttributesFromMessageContent(
  content: ConverseContentBlock,
): Record<string, AttributeValue> {
  const attributes: Record<string, AttributeValue> = {};

  if (isConverseTextContent(content)) {
    attributes[SemanticConventions.MESSAGE_CONTENT_TYPE] = "text";
    attributes[SemanticConventions.MESSAGE_CONTENT_TEXT] = content.text;
  } else if (isConverseImageContent(content)) {
    attributes[SemanticConventions.MESSAGE_CONTENT_TYPE] = "image";
    if (content.image.format) {
      attributes[`${SemanticConventions.MESSAGE_CONTENT_IMAGE}.image.format`] =
        content.image.format;
    }
    if (content.image.source.bytes) {
      // Convert bytes to base64 data URL for consistent representation
      const base64 = Buffer.from(content.image.source.bytes).toString("base64");
      const mimeType = `image/${content.image.format}`;
      attributes[
        `${SemanticConventions.MESSAGE_CONTENT_IMAGE}.${SemanticConventions.IMAGE_URL}`
      ] = `data:${mimeType};base64,${base64}`;
    }
  } else if (isConverseToolResultContent(content)) {
    attributes[SemanticConventions.MESSAGE_CONTENT_TYPE] = "tool_result";
    attributes[`${SemanticConventions.MESSAGE_CONTENT_TEXT}.tool_result.tool_use_id`] = content.toolResult.toolUseId;
    if (content.toolResult.status) {
      attributes[`${SemanticConventions.MESSAGE_CONTENT_TEXT}.tool_result.status`] = content.toolResult.status;
    }
    // Process nested content in tool result
    for (const [index, nestedContent] of content.toolResult.content.entries()) {
      const nestedAttributes = getAttributesFromMessageContent(nestedContent);
      for (const [key, value] of Object.entries(nestedAttributes)) {
        attributes[`${SemanticConventions.MESSAGE_CONTENT_TEXT}.tool_result.content.${index}.${key}`] = value;
      }
    }
  }

  return attributes;
}

/**
 * Processes multiple messages and sets attributes on span with proper indexing
 */
export function processMessages(
  span: Span,
  messages: ConverseMessage[],
  baseKey: string,
): void {
  for (const [index, message] of messages.entries()) {
    const messageAttributes = getAttributesFromMessage(message);
    for (const [key, value] of Object.entries(messageAttributes)) {
      setSpanAttribute(span, `${baseKey}.${index}.${key}`, value);
    }
  }
}
