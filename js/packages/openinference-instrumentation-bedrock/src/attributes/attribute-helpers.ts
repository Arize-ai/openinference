import type { ContentBlock, Message, SystemContentBlock } from "@aws-sdk/client-bedrock-runtime";
import type { Attributes, AttributeValue, Span } from "@opentelemetry/api";
import { diag } from "@opentelemetry/api";

import {
  isObjectWithStringKeys,
  safelyJSONStringify,
  withSafety,
} from "@arizeai/openinference-core";
import {
  LLMProvider,
  LLMSystem,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import {
  isConverseImageContent,
  isConverseReasoningContent,
  isConverseTextContent,
  isConverseToolResultContent,
  isConverseToolUseContent,
} from "../types/bedrock-types";
import { formatImageUrl } from "./invoke-model-helpers";

type ProcessableMessage = Omit<Message, "role"> & { role: string | undefined };

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
  if (modelId.includes("openai")) return LLMSystem.OPENAI;
  return LLMSystem.AMAZON;
}

export function setBasicSpanAttributes(span: Span, llm_system: LLMSystem) {
  setSpanAttribute(span, SemanticConventions.LLM_PROVIDER, LLMProvider.AWS);

  setSpanAttribute(span, SemanticConventions.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.LLM);

  setSpanAttribute(span, SemanticConventions.LLM_SYSTEM, llm_system);
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
): ProcessableMessage[] {
  const aggregated: ProcessableMessage[] = [];

  if (systemPrompts.length > 0) {
    aggregated.push({
      role: "system",
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
export const toBase64Bytes = withSafety({
  fn: (bytes: Uint8Array | Buffer | string): string | undefined => {
    if (typeof bytes === "string") {
      return bytes;
    }
    if (Buffer.isBuffer(bytes)) {
      return bytes.toString("base64");
    }
    if (bytes instanceof Uint8Array) {
      return Buffer.from(bytes).toString("base64");
    }
    diag.warn("Unsupported bytes type encountered");
    return undefined;
  },
  onError: (error) => {
    diag.warn("Failed to convert bytes to base64", error);
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
export function getAttributesFromMessageContent(content: ContentBlock): Attributes {
  const attributes: Attributes = {};

  if (isConverseTextContent(content)) {
    attributes[SemanticConventions.MESSAGE_CONTENT_TYPE] = "text";
    attributes[SemanticConventions.MESSAGE_CONTENT_TEXT] = content.text;
  } else if (isConverseImageContent(content)) {
    attributes[SemanticConventions.MESSAGE_CONTENT_TYPE] = "image";
    const base64 = toBase64Bytes(content.image.source.bytes);
    if (base64) {
      const mimeType = `image/${content.image.format}`;
      attributes[`${SemanticConventions.MESSAGE_CONTENT_IMAGE}.${SemanticConventions.IMAGE_URL}`] =
        formatImageUrl({
          type: "base64",
          data: base64,
          media_type: mimeType,
        });
    }
    // Add format attribute for image content
    attributes[`${SemanticConventions.MESSAGE_CONTENT_IMAGE}.format`] = content.image.format;
  } else if (isConverseReasoningContent(content)) {
    attributes[SemanticConventions.MESSAGE_CONTENT_TYPE] = "reasoning";
    const { reasoningText, redactedContent } = content.reasoningContent;
    if (reasoningText?.text) {
      attributes[SemanticConventions.MESSAGE_CONTENT_TEXT] = reasoningText.text;
    }
    if (reasoningText?.signature) {
      attributes[SemanticConventions.MESSAGE_CONTENT_SIGNATURE] = reasoningText.signature;
    }
    if (redactedContent) {
      const base64 = toBase64Bytes(redactedContent);
      if (base64) {
        attributes[SemanticConventions.MESSAGE_CONTENT_DATA] = base64;
      }
    }
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
export function getAttributesFromMessage(message: ProcessableMessage): Attributes {
  const attributes: Attributes = {};

  if (message.role) {
    attributes[SemanticConventions.MESSAGE_ROLE] = message.role;
  }

  if (message.content) {
    // Check if this is a simple single text content case
    if (message.content.length === 1 && isConverseTextContent(message.content[0])) {
      // Use simple format for single text content
      attributes[SemanticConventions.MESSAGE_CONTENT] = message.content[0].text;
    } else {
      // Use complex format for multi-modal or multi-content messages
      let toolCallIndex = 0;

      for (const [index, content] of message.content.entries()) {
        // Process content as our custom types for attribute extraction
        const contentAttributes = getAttributesFromMessageContent(content);
        for (const key in contentAttributes) {
          attributes[`${SemanticConventions.MESSAGE_CONTENTS}.${index}.${key}`] =
            contentAttributes[key];
        }

        // Handle tool calls at the message level using proper semantic conventions
        if (isConverseToolUseContent(content)) {
          const toolCallPrefix = `${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}`;
          attributes[`${toolCallPrefix}.${SemanticConventions.TOOL_CALL_ID}`] =
            content.toolUse.toolUseId;
          attributes[`${toolCallPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`] =
            content.toolUse.name;
          const argsJson = safelyJSONStringify(content.toolUse.input);
          if (argsJson) {
            attributes[
              `${toolCallPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
            ] = argsJson;
          }
          toolCallIndex++;
        }
      }
    }
  }

  return attributes;
}

/**
 * Converts the text and JSON blocks of a toolResult to one message content string
 * Image, document and video blocks are not recorded
 */
function getToolResultContent(content: unknown): string | undefined {
  if (!Array.isArray(content)) {
    return undefined;
  }
  const parts: string[] = [];
  for (const block of content) {
    if (isObjectWithStringKeys(block)) {
      if (typeof block.text === "string") {
        parts.push(block.text);
      }
      if ("json" in block) {
        const json = safelyJSONStringify(block.json);
        if (json != null) {
          parts.push(json);
        }
      }
    }
  }
  return parts.length > 0 ? parts.join("\n\n") : undefined;
}

/**
 * Computes the attributes of each message to record, with a separate tool message for each
 * toolResult block. A Converse message can carry several toolResult blocks (parallel tool
 * calls), while a message has one tool_call_id and one content.
 */
function getAttributesFromMessages(messages: ProcessableMessage[]): Attributes[] {
  const result: Attributes[] = [];
  for (const message of messages) {
    const toolResultMessages: Attributes[] = [];
    const otherContent: ContentBlock[] = [];
    for (const content of message.content ?? []) {
      if (isConverseToolResultContent(content)) {
        const toolMessage: Attributes = {
          [SemanticConventions.MESSAGE_ROLE]: "tool",
          [SemanticConventions.MESSAGE_TOOL_CALL_ID]: content.toolResult.toolUseId,
        };
        const toolContent = getToolResultContent(content.toolResult.content);
        if (toolContent != null) {
          toolMessage[SemanticConventions.MESSAGE_CONTENT] = toolContent;
        }
        toolResultMessages.push(toolMessage);
      } else {
        otherContent.push(content);
      }
    }
    if (toolResultMessages.length === 0) {
      result.push(getAttributesFromMessage(message));
      continue;
    }
    // A message whose only content was tool results is fully represented by the tool messages
    if (otherContent.length > 0) {
      result.push(getAttributesFromMessage({ ...message, content: otherContent }));
    }
    result.push(...toolResultMessages);
  }
  return result;
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
  messages: ProcessableMessage[];
  baseKey:
    | typeof SemanticConventions.LLM_INPUT_MESSAGES
    | typeof SemanticConventions.LLM_OUTPUT_MESSAGES;
}): void {
  for (const [index, messageAttributes] of getAttributesFromMessages(messages).entries()) {
    for (const [key, value] of Object.entries(messageAttributes)) {
      setSpanAttribute(span, `${baseKey}.${index}.${key}`, value);
    }
  }
}

/**
 * Extracts clean model name from full Bedrock model ID
 * Removes the optional region prefix and vendor prefix, preserving the remaining
 * model string (which may itself contain dots, e.g. "deepseek.v3.2" -> "v3.2").
 *
 * Handles both standard and cross-region inference model IDs, plus ARNs:
 *   - Standard:      "anthropic.claude-3-sonnet-20240229-v1:0"
 *   - Cross-region:  "us.anthropic.claude-haiku-4-5-20251001-v1:0"
 *   - ARN:           "arn:aws:bedrock:...:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0"
 *   - Dotted model:  "deepseek.v3.2" -> "v3.2"
 *
 * @param modelId The full Bedrock model identifier
 * @returns {string} The cleaned model name (e.g., "claude-3-sonnet-20240229" from "anthropic.claude-3-sonnet-20240229-v1:0")
 */
export function extractModelName(modelId: string): string {
  // For ARNs, use only the resource portion after the last "/".
  const slashIndex = modelId.lastIndexOf("/");
  const id = slashIndex >= 0 ? modelId.substring(slashIndex + 1) : modelId;

  const parts = id.split(".");
  if (parts.length <= 1) {
    return modelId;
  }

  // Known region prefixes used by cross-region inference profiles.
  const regionPrefixes = new Set(["us", "eu", "ap", "global", "apac"]);

  let startIndex = 0;
  // Strip the optional region prefix.
  if (regionPrefixes.has(parts[0])) {
    startIndex = 1;
  }
  // Strip the vendor prefix, leaving the remaining model string (dots included).
  if (startIndex + 1 < parts.length) {
    startIndex += 1;
  }

  const modelPart = parts.slice(startIndex).join(".");

  // Anthropic model IDs carry a "-vN:0" version suffix that we strip for readability.
  if (id.includes("anthropic")) {
    const versionIndex = modelPart.indexOf("-v");
    if (versionIndex > 0) {
      return modelPart.substring(0, versionIndex);
    }
  }
  return modelPart;
}
