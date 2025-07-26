/**
 * TypeScript type definitions for AWS Bedrock Runtime API structures
 *
 * This file imports official AWS SDK types wherever possible and only defines
 * custom types for structures not exposed by the SDK.
 */

// Import official AWS SDK types
import {
  // Core content and message types from SDK
  ContentBlock,
  Message,
  SystemContentBlock,
  ConversationRole,

  // Tool-related types
  Tool,
  ToolUseBlock,
  ToolResultBlock,
  ToolChoice,
  ToolInputSchema,

  // Configuration types
  InferenceConfiguration,

  // Streaming types
  ContentBlockStart,
  ContentBlockDelta,
  ContentBlockDeltaEvent,
  ContentBlockStartEvent,
  ContentBlockStopEvent,
} from "@aws-sdk/client-bedrock-runtime";
import { isObjectWithStringKeys } from "@arizeai/openinference-core";

// Re-export AWS SDK types for convenience
export {
  ContentBlock,
  Message,
  SystemContentBlock,
  ConversationRole,
  Tool,
  ToolUseBlock,
  ToolResultBlock,
  ToolChoice,
  ToolInputSchema,
  InferenceConfiguration,
  ContentBlockStart,
  ContentBlockDelta,
  ContentBlockDeltaEvent,
  ContentBlockStartEvent,
  ContentBlockStopEvent,
};

// Custom types that extend or aren't available in the SDK

// Legacy InvokeModel API types (not fully exposed in new SDK versions)
export interface InvokeModelRequestBody {
  anthropic_version: string;
  max_tokens: number;
  messages: BedrockMessage[];
  tools?: ToolDefinition[];
  system?: string;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stop_sequences?: string[];
}

export interface InvokeModelResponseBody {
  id: string;
  type: "message";
  role: "assistant";
  content: (TextContent | ToolUseContent)[];
  model: string;
  stop_reason: string;
  stop_sequence?: string;
  usage: UsageInfo;
}

// Legacy message format for InvokeModel (different from Converse Message)
export interface BedrockMessage {
  role: "user" | "assistant";
  content: MessageContent;
}

export type MessageContent =
  | string
  | (TextContent | ImageContent | ToolUseContent | ToolResultContent)[];

// Content types for legacy InvokeModel API
export interface TextContent {
  type: "text";
  text: string;
}

export interface ImageSource {
  type: "base64";
  media_type: string;
  data: string;
}

export interface ImageContent {
  type: "image";
  source: ImageSource;
}

export interface ToolUseContent {
  type: "tool_use";
  id: string;
  name: string;
  input: Record<string, unknown>;
}

export interface ToolResultContent {
  type: "tool_result";
  tool_use_id: string;
  content: string;
}

// Tool definition for legacy InvokeModel API
export interface ToolDefinition {
  name: string;
  description: string;
  input_schema: {
    type: "object";
    properties: Record<string, unknown>;
    required: string[];
  };
}

// Usage information
export interface UsageInfo {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
}

// Content blocks for Converse API (map to SDK ContentBlock structure)
// These are used for type guards since AWS SDK ContentBlock has complex union types
export type ConverseContentBlock =
  | ConverseTextContent
  | ConverseImageContent
  | ConverseToolUseContent
  | ConverseToolResultContent;

export interface ConverseTextContent {
  text: string;
}

export interface ConverseImageContent {
  image: {
    format: "png" | "jpeg" | "gif" | "webp";
    source: {
      bytes: Uint8Array;
    };
  };
}

export interface ConverseToolUseContent {
  toolUse: {
    toolUseId: string;
    name: string;
    input: Record<string, unknown>;
  };
}

export interface ConverseToolResultContent {
  toolResult: {
    toolUseId: string;
    content: ConverseContentBlock[];
    status?: "success" | "error";
  };
}

// Type guards for legacy InvokeModel content types
export function isTextContent(content: unknown): content is TextContent {
  return (
    content !== null &&
    typeof content === "object" &&
    content !== null &&
    "type" in content &&
    content.type === "text" &&
    "text" in content &&
    typeof content.text === "string"
  );
}

export function isImageContent(content: unknown): content is ImageContent {
  return (
    content !== null &&
    typeof content === "object" &&
    content !== null &&
    "type" in content &&
    content.type === "image" &&
    "source" in content &&
    content.source !== null &&
    typeof content.source === "object" &&
    "type" in content.source &&
    content.source.type === "base64"
  );
}

export function isToolUseContent(content: unknown): content is ToolUseContent {
  return (
    content !== null &&
    typeof content === "object" &&
    content !== null &&
    "type" in content &&
    content.type === "tool_use" &&
    "name" in content &&
    typeof content.name === "string"
  );
}

export function isToolResultContent(
  content: unknown,
): content is ToolResultContent {
  return (
    content !== null &&
    typeof content === "object" &&
    content !== null &&
    "type" in content &&
    content.type === "tool_result" &&
    "tool_use_id" in content &&
    typeof content.tool_use_id === "string"
  );
}

// Type guards for Converse content blocks
// These are needed because AWS SDK ContentBlock has complex union types that don't match our processing needs
export function isConverseTextContent(
  content: unknown,
): content is ConverseTextContent {
  return (
    content !== null &&
    typeof content === "object" &&
    content !== null &&
    "text" in content &&
    typeof content.text === "string"
  );
}

export function isConverseImageContent(
  content: unknown,
): content is ConverseImageContent {
  return (
    content !== null &&
    typeof content === "object" &&
    content !== null &&
    "image" in content &&
    content.image !== null &&
    typeof content.image === "object" &&
    "source" in content.image &&
    content.image.source !== null &&
    typeof content.image.source === "object" &&
    "bytes" in content.image.source &&
    content.image.source.bytes instanceof Uint8Array
  );
}

export function isConverseToolUseContent(
  content: unknown,
): content is ConverseToolUseContent {
  return (
    content !== null &&
    typeof content === "object" &&
    content !== null &&
    "toolUse" in content &&
    content.toolUse !== null &&
    typeof content.toolUse === "object" &&
    "name" in content.toolUse &&
    typeof content.toolUse.name === "string" &&
    "toolUseId" in content.toolUse &&
    typeof content.toolUse.toolUseId === "string"
  );
}

export function isConverseToolResultContent(
  content: unknown,
): content is ConverseToolResultContent {
  return (
    isObjectWithStringKeys(content) &&
    "toolResult" in content &&
    isObjectWithStringKeys(content.toolResult) &&
    "toolUseId" in content.toolResult &&
    typeof content.toolResult.toolUseId === "string"
  );
}
