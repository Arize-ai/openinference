import { Span, AttributeValue } from "@opentelemetry/api";

/**
 * Sets a span attribute only if the value is not null, undefined, or empty string
 * Provides null-safe attribute setting for OpenTelemetry spans
 */
export function setSpanAttribute(
  span: Span,
  key: string,
  value: AttributeValue | null | undefined,
): void {
  if (value !== undefined && value !== null && value !== "") {
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

// ========================================================================
// CONVERSE API HELPER FUNCTIONS
// ========================================================================

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
 * Aggregates multiple system prompts into a single string with space separation
 */
export function aggregateSystemPrompts(systemPrompts: SystemPrompt[]): string {
  return systemPrompts
    .map((prompt) => prompt.text || "")
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
 * Generator function to extract attributes from a single message
 */
export function* getAttributesFromMessage(
  message: ConverseMessage,
): Generator<[string, AttributeValue]> {
  if (message.role) {
    yield ["message.role", message.role];
  }

  if (message.content) {
    for (const [index, content] of message.content.entries()) {
      for (const [key, value] of getAttributesFromMessageContent(content)) {
        yield [`message.contents.${index}.${key}`, value];
      }
    }
  }
}

/**
 * Generator function to extract attributes from message content
 */
export function* getAttributesFromMessageContent(
  content: ConverseContentBlock,
): Generator<[string, AttributeValue]> {
  if (isConverseTextContent(content)) {
    yield ["message_content.type", "text"];
    yield ["message_content.text", content.text];
  } else if (isConverseImageContent(content)) {
    yield ["message_content.type", "image"];
    if (content.image.format) {
      yield ["message_content.image.image.format", content.image.format];
    }
    if (content.image.source.bytes) {
      // Convert bytes to base64 data URL for consistent representation
      const base64 = Buffer.from(content.image.source.bytes).toString("base64");
      const mimeType = `image/${content.image.format}`;
      yield [
        "message_content.image.image.url",
        `data:${mimeType};base64,${base64}`,
      ];
    }
  } else if (isConverseToolUseContent(content)) {
    yield ["message_content.type", "tool_use"];
    yield ["message_content.tool_use.id", content.toolUse.toolUseId];
    yield ["message_content.tool_use.name", content.toolUse.name];
    if (content.toolUse.input) {
      yield [
        "message_content.tool_use.input",
        JSON.stringify(content.toolUse.input),
      ];
    }
  } else if (isConverseToolResultContent(content)) {
    yield ["message_content.type", "tool_result"];
    yield [
      "message_content.tool_result.tool_use_id",
      content.toolResult.toolUseId,
    ];
    if (content.toolResult.status) {
      yield ["message_content.tool_result.status", content.toolResult.status];
    }
    // Process nested content in tool result
    for (const [index, nestedContent] of content.toolResult.content.entries()) {
      for (const [key, value] of getAttributesFromMessageContent(
        nestedContent,
      )) {
        yield [`message_content.tool_result.content.${index}.${key}`, value];
      }
    }
  }
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
    for (const [key, value] of getAttributesFromMessage(message)) {
      setSpanAttribute(span, `${baseKey}.${index}.${key}`, value);
    }
  }
}
