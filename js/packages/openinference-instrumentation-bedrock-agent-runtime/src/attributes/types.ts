/**
 * Represents a message with optional role, content, and tool call information.
 */
export interface Message {
  role?: string;
  content?: string;
  contents?: MessageContent[];
  tool_call_id?: string;
  tool_calls?: ToolCall[];
  [key: string]: unknown;
}

/**
 * Represents the token count for a message, including prompt, completion, and total tokens.
 */
export interface TokenCount {
  prompt?: number;
  completion?: number;
  total?: number;
  [key: string]: unknown;
}

/**
 * Represents the function called by a tool, including its name and arguments.
 */
export interface ToolCallFunction {
  name?: string;
  arguments?: string | Record<string, unknown>;
  [key: string]: unknown;
}

/**
 * Represents a tool call, including its ID and the function invoked.
 */
export interface ToolCall {
  id?: string;
  function?: ToolCallFunction;
  [key: string]: unknown;
}

/**
 * Union type for message content, which can be text or image.
 */
export type MessageContent = TextMessageContent | ImageMessageContent;

/**
 * Represents an image with a URL and optional additional properties.
 */
export interface Image {
  url: string;
  [key: string]: string;
}

/**
 * Represents a text message content block.
 */
export interface TextMessageContent {
  type: "text";
  text: string;
  [key: string]: string;
}

/**
 * Represents an image message content block.
 */
export interface ImageMessageContent {
  type: "image";
  image: Image;
  [key: string]: Image | string;
}

export interface ParsedMessage {
  role?: string;
  content?: string;
  [key: string]: unknown;
}

export interface ParsedInput {
  system?: string;
  type?: string;
  messages?: ParsedMessage[];
  [key: string]: unknown;
}

export interface DocumentReference {
  metadata?: Record<string, unknown> & {
    ["x-amz-bedrock-kb-chunk-id"]?: string;
  };
  content?: {
    text?: string;
    type?: string;
  };
  score?: number;
  location?: Record<string, unknown>;
  [key: string]: unknown;
}
