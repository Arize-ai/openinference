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

  // Converse Stream types
  ConverseStreamCommand,
  ConverseStreamCommandInput,
  ConverseStreamCommandOutput,
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
  ConverseStreamCommand,
  ConverseStreamCommandInput,
  ConverseStreamCommandOutput,
};

// Custom types that extend or aren't available in the SDK

// Flexible type for multi-provider support with runtime validation
export type InvokeModelRequestBody = Record<string, unknown>;
export type InvokeModelResponseBody = Record<string, unknown>;
export type InvocationParameters = Record<string, unknown>;

// Converse Stream event data types using AWS SDK types where possible
// This is a discriminated union of all possible streaming event types
export interface ConverseStreamEventData {
  // The event type discriminator
  type?: string;

  // Message lifecycle events
  messageStart?: {
    role: ConversationRole;
  };
  messageStop?: {
    stopReason: string;
    additionalModelResponseFields?: Record<string, unknown>;
  };

  // Content block events using AWS SDK types
  contentBlockStart?: {
    start: ContentBlockStart;
    contentBlockIndex: number;
  };
  contentBlockDelta?: {
    delta: ContentBlockDelta;
    contentBlockIndex: number;
  };
  contentBlockStop?: {
    contentBlockIndex: number;
  };

  // Raw event fields from actual streaming response
  content_block?: {
    type: string;
    id?: string;
    name?: string;
    input?: Record<string, unknown>;
  };
  index?: number;
  delta?: {
    type: string;
    text?: string;
  };
  partial_json?: string;

  // Metadata and usage events
  metadata?: {
    usage: {
      inputTokens: number;
      outputTokens: number;
      totalTokens: number;
    };
    metrics: {
      latencyMs: number;
    };
  };
}

// Stream processing state for converse streaming
export interface ConverseStreamProcessingState {
  outputText: string;
  toolCalls: Array<{
    id: string;
    name: string;
    input: Record<string, unknown>;
    partialJsonInput?: string; // Accumulates partial JSON chunks
  }>;
  usage: {
    inputTokens?: number;
    outputTokens?: number;
    totalTokens?: number;
  };
  stopReason?: string;
}

// Extended conversation role type to support additional roles like "tool" and "system" for Mistral
export type ExtendedConversationRole = ConversationRole | "tool" | "system";

// Legacy message format for InvokeModel (different from Converse Message)
export interface BedrockMessage {
  role: ExtendedConversationRole;
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

/**
 * Comprehensive usage attributes normalized from various provider response formats
 * Includes all token count types that may be present across different model providers
 * All fields are optional - undefined means the token count was not provided by the model
 */
export interface UsageAttributes {
  /** Input/prompt tokens consumed */
  input_tokens?: number;
  /** Output/completion tokens generated */
  output_tokens?: number;
  /** Total tokens (input + output), if provided by the model */
  total_tokens?: number;
  /** Cache read tokens for prompt caching features */
  cache_read_input_tokens?: number;
  /** Cache write tokens for prompt caching features */
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
      bytes: Uint8Array | string | Buffer | {type: "Buffer", data: number[]}; // Support various formats
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
    "image" in content &&
    content.image !== null &&
    typeof content.image === "object" &&
    "source" in content.image &&
    content.image.source !== null &&
    typeof content.image.source === "object" &&
    "bytes" in content.image.source &&
    (content.image.source.bytes instanceof Uint8Array || 
     typeof content.image.source.bytes === "string" ||
     Buffer.isBuffer(content.image.source.bytes) ||
     (typeof content.image.source.bytes === "object" && 
      content.image.source.bytes !== null &&
      "type" in content.image.source.bytes && 
      content.image.source.bytes.type === "Buffer"))
  );
}

export function isConverseToolUseContent(
  content: unknown,
): content is ConverseToolUseContent {
  return (
    content !== null &&
    typeof content === "object" &&
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

// Type guards for Converse streaming types
export function isValidConverseStreamEventData(
  data: unknown,
): data is ConverseStreamEventData {
  return isObjectWithStringKeys(data);
}

export function isConverseMessageStartEvent(
  data: ConverseStreamEventData,
): boolean {
  return data.messageStart !== undefined;
}

export function isConverseMessageStopEvent(
  data: ConverseStreamEventData,
): boolean {
  return data.messageStop !== undefined;
}

export function isConverseContentBlockStartEvent(
  data: ConverseStreamEventData,
): boolean {
  return data.contentBlockStart !== undefined;
}

export function isConverseContentBlockDeltaEvent(
  data: ConverseStreamEventData,
): boolean {
  return data.contentBlockDelta !== undefined;
}

export function isConverseContentBlockStopEvent(
  data: ConverseStreamEventData,
): boolean {
  return data.contentBlockStop !== undefined;
}

export function isConverseMetadataEvent(
  data: ConverseStreamEventData,
): boolean {
  return data.metadata !== undefined;
}
