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
import { diag } from "@opentelemetry/api";

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

/**
 * Arbitrary request body for Bedrock InvokeModel-style APIs.
 * Provider-specific payloads vary; callers should validate at runtime.
 */
export type InvokeModelRequestBody = Record<string, unknown>;
/**
 * Arbitrary response body for Bedrock InvokeModel-style APIs.
 * Provider-specific payloads vary; consumers should normalize as needed.
 */
export type InvokeModelResponseBody = Record<string, unknown>;
/** Generic invocation parameters bag used across providers. */
export type InvocationParameters = Record<string, unknown>;

// Converse Stream raw event data as emitted by the SDK and Bedrock service
// This is a flexible shape; it is normalized by toNormalizedConverseStreamEvent below
/**
 * Raw Converse streaming event structure capturing the superset of possible
 * fields. Use toNormalizedConverseStreamEvent to map to a discriminated union
 * for downstream processing.
 */
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

/** In-memory state accumulated while consuming a Converse stream. */
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
  /** Map of contentBlockIndex -> toolUseId for correlating input chunks */
  toolUseIdByIndex?: Record<number, string>;
}

/**
 * Extended role type used by some providers (e.g., Mistral) in legacy InvokeModel flows.
 */
export type ExtendedConversationRole = ConversationRole | "tool" | "system";

/** Legacy InvokeModel message format (different from Converse messages). */
export interface BedrockMessage {
  role: ExtendedConversationRole;
  content: MessageContent;
}

/** Union of legacy InvokeModel message content types. */
export type MessageContent =
  | string
  | (TextContent | ImageContent | ToolUseContent | ToolResultContent)[];

/** Text content block used by legacy InvokeModel APIs. */
export interface TextContent {
  type: "text";
  text: string;
}

/** Base64-encoded image source used by legacy InvokeModel APIs. */
export interface ImageSource {
  type: "base64";
  media_type: string;
  data: string;
}

/** Image content block used by legacy InvokeModel APIs. */
export interface ImageContent {
  type: "image";
  source: ImageSource;
}

/** Tool call content used by legacy InvokeModel APIs. */
export interface ToolUseContent {
  type: "tool_use";
  id: string;
  name: string;
  input: Record<string, unknown>;
}

/** Tool result content used by legacy InvokeModel APIs. */
export interface ToolResultContent {
  type: "tool_result";
  tool_use_id: string;
  content: string;
}

/** Tool/function definition used by legacy InvokeModel APIs. */
export interface ToolDefinition {
  name: string;
  description: string;
  input_schema: {
    type: "object";
    properties: Record<string, unknown>;
    required: string[];
  };
}

/** Provider-reported usage info (legacy InvokeModel flavor). */
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
// These are used for type guards since AWS SDK ContentBlock has complex unions
export type ConverseContentBlock =
  | ConverseTextContent
  | ConverseImageContent
  | ConverseToolUseContent
  | ConverseToolResultContent;

/** Text content block in Converse API messages. */
export interface ConverseTextContent {
  text: string;
}

/** Image content block in Converse API messages. */
export interface ConverseImageContent {
  image: {
    format: "png" | "jpeg" | "gif" | "webp";
    source: {
      bytes: Uint8Array | string | Buffer | { type: "Buffer"; data: number[] }; // Support various formats
    };
  };
}

/** Tool call content block in Converse API messages. */
export interface ConverseToolUseContent {
  toolUse: {
    toolUseId: string;
    name: string;
    input: Record<string, unknown>;
  };
}

/** Tool result content block in Converse API messages. */
export interface ConverseToolResultContent {
  toolResult: {
    toolUseId: string;
    content: ConverseContentBlock[];
    status?: "success" | "error";
  };
}

/** Type guard for legacy text content blocks. */
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

/** Type guard for legacy image content blocks. */
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

/** Type guard for legacy tool call content blocks. */
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

/** Type guard for legacy tool result content blocks. */
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

/** Type guard for Converse text content. */
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

/** Type guard for Converse image content. */
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
      Buffer.isBuffer(content.image.source.bytes))
  );
}

/** Type guard for Converse tool call content. */
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

/** Type guard for Converse tool result content. */
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

/**
 * Loose type guard for Converse streaming events.
 * Ensures we have an object and at least one known event marker.
 * Exhaustive discrimination happens in toNormalizedConverseStreamEvent.
 */
export function isValidConverseStreamEventData(
  data: unknown,
): data is ConverseStreamEventData {
  if (!isObjectWithStringKeys(data)) {
    return false;
  }

  return (
    "messageStart" in data ||
    "messageStop" in data ||
    "contentBlockStart" in data ||
    "contentBlockDelta" in data ||
    "contentBlockStop" in data ||
    "metadata" in data ||
    // raw-wire style events expose `type`
    "type" in data
  );
}

/**
 * Discriminated union of normalized Converse streaming events.
 * Produced by toNormalizedConverseStreamEvent for safe downstream handling.
 */
export type NormalizedConverseStreamEvent =
  | { kind: "messageStart" }
  | { kind: "messageStop"; stopReason?: string }
  | { kind: "textDelta"; text: string }
  | {
      kind: "toolUseStart";
      id: string;
      name: string;
      contentBlockIndex?: number;
    }
  | {
      kind: "toolUseInputChunk";
      chunk: string;
      contentBlockIndex?: number;
      id?: string;
    }
  | {
      kind: "metadata";
      usage: {
        inputTokens?: number;
        outputTokens?: number;
        totalTokens?: number;
      };
    };

/**
 * Normalizes raw Converse streaming events into a discriminated union.
 * Handles both structured SDK events and raw-wire events.
 */
export function toNormalizedConverseStreamEvent(
  e: ConverseStreamEventData,
): NormalizedConverseStreamEvent | undefined {
  // Structured SDK events
  if (e.messageStart) return { kind: "messageStart" };
  if (e.messageStop)
    return { kind: "messageStop", stopReason: e.messageStop.stopReason };
  if (e.contentBlockDelta?.delta?.text)
    return { kind: "textDelta", text: e.contentBlockDelta.delta.text };
  if (
    e.contentBlockStart?.start?.toolUse?.toolUseId &&
    e.contentBlockStart.start.toolUse.name
  ) {
    return {
      kind: "toolUseStart",
      id: e.contentBlockStart.start.toolUse.toolUseId,
      name: e.contentBlockStart.start.toolUse.name,
      contentBlockIndex: e.contentBlockStart.contentBlockIndex,
    };
  }
  if (e.contentBlockDelta?.delta?.toolUse?.input !== undefined) {
    return {
      kind: "toolUseInputChunk",
      chunk: String(e.contentBlockDelta.delta.toolUse.input),
      contentBlockIndex: e.contentBlockDelta.contentBlockIndex,
    };
  }

  // Raw wire events
  if (
    e.type === "content_block_delta" &&
    e.delta?.type === "text_delta" &&
    e.delta.text
  ) {
    return { kind: "textDelta", text: e.delta.text };
  }
  if (
    e.type === "content_block_start" &&
    e.content_block?.type === "tool_use" &&
    e.content_block.id &&
    e.content_block.name
  ) {
    return {
      kind: "toolUseStart",
      id: e.content_block.id,
      name: e.content_block.name,
    };
  }
  if (e.type === "input_json_delta" && e.partial_json) {
    return { kind: "toolUseInputChunk", chunk: e.partial_json };
  }
  if (e.metadata?.usage) {
    return {
      kind: "metadata",
      usage: {
        inputTokens: e.metadata.usage.inputTokens,
        outputTokens: e.metadata.usage.outputTokens,
        totalTokens: e.metadata.usage.totalTokens,
      },
    };
  }

  // Exhaustive warning for unexpected/unhandled event shape
  diag.warn("Encountered unexpected Converse stream event shape; dropping.", e);
}
