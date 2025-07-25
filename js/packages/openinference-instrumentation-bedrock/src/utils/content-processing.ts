/**
 * Content processing utilities for AWS Bedrock instrumentation
 *
 * Handles extraction and conversion of various content types including:
 * - Text content
 * - Multi-modal content (images)
 * - Tool use and tool result content
 */

import {
  MessageContent,
  TextContent,
  ToolUseContent,
  ToolResultContent,
  ImageSource,
  isTextContent,
  isImageContent,
  isToolUseContent,
  isToolResultContent,
} from "../types/bedrock-types";

// Content formatting constants
const TOOL_RESULT_PREFIX = "[Tool Result: ";
const TOOL_CALL_PREFIX = "[Tool Call: ";
const CONTENT_SUFFIX = "]";

/**
 * Extracts text content from various message content formats
 * Supports both string content and complex content arrays with multi-modal data
 */
export function extractTextFromContent(content: MessageContent): string {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    const textParts: string[] = [];

    content.forEach((block) => {
      if (isTextContent(block)) {
        textParts.push(block.text);
      } else if (isToolResultContent(block)) {
        textParts.push(`${TOOL_RESULT_PREFIX}${block.content}${CONTENT_SUFFIX}`);
      } else if (isToolUseContent(block)) {
        textParts.push(`${TOOL_CALL_PREFIX}${block.name}${CONTENT_SUFFIX}`);
      } else if (isImageContent(block)) {
        // Handle image content in OpenInference format: data:{media_type};base64,{data}
        const imageUrl = formatImageUrl(block.source);
        if (imageUrl) {
          textParts.push(imageUrl);
        }
      }
    });

    return textParts.join(" ");
  }

  return "";
}

/**
 * Formats image source data into OpenInference data URL format
 * Converts Bedrock image source to: data:{media_type};base64,{data}
 */
export function formatImageUrl(source: ImageSource): string {
  if (source.type === "base64" && source.data && source.media_type) {
    return `data:${source.media_type};base64,${source.data}`;
  }
  return "";
}

/**
 * Extracts tool use content blocks from message content
 * Returns array of tool use blocks for tool call attribute processing
 */
export function extractToolUseBlocks(
  content: MessageContent,
): ToolUseContent[] {
  if (typeof content === "string" || !Array.isArray(content)) {
    return [];
  }

  return content.filter(isToolUseContent);
}

/**
 * Extracts tool result content blocks from message content
 * Returns array of tool result blocks for tool result attribute processing
 */
export function extractToolResultBlocks(
  content: MessageContent,
): ToolResultContent[] {
  if (typeof content === "string" || !Array.isArray(content)) {
    return [];
  }

  return content.filter(isToolResultContent);
}

/**
 * Extracts text content blocks from message content
 * Returns array of text blocks for structured message processing
 */
export function extractTextBlocks(content: MessageContent): TextContent[] {
  if (typeof content === "string") {
    return [{ type: "text", text: content }];
  }

  if (!Array.isArray(content)) {
    return [];
  }

  return content.filter(isTextContent);
}
