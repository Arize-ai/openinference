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
 * Extracts text content from various Bedrock message content formats
 * Supports both string content and complex content arrays with multi-modal data
 * Formats tool calls and results with descriptive prefixes for readability
 *
 * @param content The message content to extract text from (string or content block array)
 * @returns {string} Extracted and formatted text content with tool information
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
        textParts.push(
          `${TOOL_RESULT_PREFIX}${block.content}${CONTENT_SUFFIX}`,
        );
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
 * Formats Bedrock image source data into OpenInference data URL format
 * Converts Bedrock image source to standard data URL: data:{media_type};base64,{data}
 *
 * @param source The Bedrock image source containing type, data, and media type
 * @returns {string} Formatted data URL or empty string if source is invalid
 */
export function formatImageUrl(source: ImageSource): string {
  if (source.type === "base64" && source.data && source.media_type) {
    return `data:${source.media_type};base64,${source.data}`;
  }
  return "";
}

/**
 * Extracts tool use content blocks from Bedrock message content
 * Filters content array to return only tool use blocks for processing tool calls
 *
 * @param content The message content to extract tool use blocks from
 * @returns {ToolUseContent[]} Array of tool use blocks, empty if none found
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
 * Extracts tool result content blocks from Bedrock message content
 * Filters content array to return only tool result blocks for processing tool responses
 *
 * @param content The message content to extract tool result blocks from
 * @returns {ToolResultContent[]} Array of tool result blocks, empty if none found
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
 * Extracts text content blocks from Bedrock message content
 * Converts string content to text block format and filters array content for text blocks
 *
 * @param content The message content to extract text blocks from
 * @returns {TextContent[]} Array of text blocks with type and text properties
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
