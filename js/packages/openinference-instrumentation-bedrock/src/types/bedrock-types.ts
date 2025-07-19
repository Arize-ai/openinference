/**
 * TypeScript type definitions for AWS Bedrock Runtime API structures
 */

// AWS SDK imports for proper typing
import {
  ConverseCommand,
  ConverseCommandInput,
  ConverseCommandOutput,
  InferenceConfiguration,
  SystemContentBlock,
  Message as AwsConverseMessage,
  ContentBlock,
  ToolConfiguration,
} from "@aws-sdk/client-bedrock-runtime";

// Core content types
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
  input: Record<string, any>;
}

export interface ToolResultContent {
  type: "tool_result";
  tool_use_id: string;
  content: string;
}

export type MessageContent =
  | string
  | (TextContent | ImageContent | ToolUseContent | ToolResultContent)[];

// Message structures
export interface BedrockMessage {
  role: "user" | "assistant";
  content: MessageContent;
}

// Tool definitions
export interface ToolDefinition {
  name: string;
  description: string;
  input_schema: {
    type: "object";
    properties: Record<string, any>;
    required: string[];
  };
}

// Request structures
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

// Response structures
export interface UsageInfo {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
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

// Type guards for runtime validation
export function isTextContent(content: any): content is TextContent {
  return (
    content &&
    typeof content === "object" &&
    content.type === "text" &&
    typeof content.text === "string"
  );
}

export function isImageContent(content: any): content is ImageContent {
  return (
    content &&
    typeof content === "object" &&
    content.type === "image" &&
    content.source &&
    content.source.type === "base64"
  );
}

export function isToolUseContent(content: any): content is ToolUseContent {
  return (
    content &&
    typeof content === "object" &&
    content.type === "tool_use" &&
    typeof content.name === "string"
  );
}

export function isToolResultContent(
  content: any,
): content is ToolResultContent {
  return (
    content &&
    typeof content === "object" &&
    content.type === "tool_result" &&
    typeof content.tool_use_id === "string"
  );
}

// ========================================================================
// CONVERSE API TYPES (NEW)
// ========================================================================

// Converse API specific types - properly typed following AWS SDK patterns

// System prompt for Converse API (different from InvokeModel)
export interface SystemPrompt {
  text: string;
}

// Converse message format (extends AWS SDK Message but with our content types)
export interface ConverseMessage {
  role: "user" | "assistant" | "system";
  content: ConverseContentBlock[];
}

// Converse content blocks (similar to ContentBlock but our structured format)
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
    input: Record<string, any>;
  };
}

export interface ConverseToolResultContent {
  toolResult: {
    toolUseId: string;
    content: ConverseContentBlock[];
    status?: "success" | "error";
  };
}

// Converse inference configuration (different parameter names from InvokeModel)
export interface ConverseInferenceConfig {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  stopSequences?: string[];
}

// Converse tool configuration (different structure from InvokeModel tools)
export interface ConverseToolConfig {
  tools?: ConverseToolDefinition[];
  toolChoice?: {
    auto?: {};
    any?: {};
    tool?: {
      name: string;
    };
  };
}

export interface ConverseToolDefinition {
  toolSpec: {
    name: string;
    description: string;
    inputSchema: {
      json: Record<string, any>;
    };
  };
}

// Converse request body (properly typed with AWS SDK alignment)
export interface ConverseRequestBody {
  modelId: string;
  messages: ConverseMessage[];
  system?: SystemPrompt[];
  inferenceConfig?: ConverseInferenceConfig;
  toolConfig?: ConverseToolConfig;
  additionalModelRequestFields?: Record<string, any>;
  additionalModelResponseFieldPaths?: string[];
}

// Converse response body (properly typed following AWS SDK structure)
export interface ConverseResponseBody {
  responseMetadata: {
    requestId: string;
    httpStatusCode: number;
  };
  output: {
    message: {
      role: "assistant";
      content: ConverseContentBlock[];
    };
  };
  stopReason:
    | "end_turn"
    | "tool_use"
    | "max_tokens"
    | "stop_sequence"
    | "content_filtered";
  usage: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
  };
  metrics?: {
    latencyMs: number;
  };
}

// Type guards for Converse content blocks
export function isConverseTextContent(
  content: any,
): content is ConverseTextContent {
  return (
    content && typeof content === "object" && typeof content.text === "string"
  );
}

export function isConverseImageContent(
  content: any,
): content is ConverseImageContent {
  return (
    content &&
    typeof content === "object" &&
    content.image &&
    content.image.source &&
    content.image.source.bytes instanceof Uint8Array
  );
}

export function isConverseToolUseContent(
  content: any,
): content is ConverseToolUseContent {
  return (
    content &&
    typeof content === "object" &&
    content.toolUse &&
    typeof content.toolUse.name === "string" &&
    typeof content.toolUse.toolUseId === "string"
  );
}

export function isConverseToolResultContent(
  content: any,
): content is ConverseToolResultContent {
  return (
    content &&
    typeof content === "object" &&
    content.toolResult &&
    typeof content.toolResult.toolUseId === "string"
  );
}
