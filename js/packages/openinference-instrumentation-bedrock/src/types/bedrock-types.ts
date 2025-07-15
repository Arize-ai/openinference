/**
 * TypeScript type definitions for AWS Bedrock Runtime API structures
 */

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
