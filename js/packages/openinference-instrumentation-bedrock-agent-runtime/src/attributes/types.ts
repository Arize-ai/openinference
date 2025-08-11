/**
 * Represents a message with optional role, content, and tool call information.
 */
export interface Message {
  role?: string;
  content?: string;
  contents?: MessageContent[];
  tool_call_id?: string;
  tool_calls?: ToolCall[];
  [key: string]: any;
}

/**
 * Represents the token count for a message, including prompt, completion, and total tokens.
 */
export interface TokenCount {
  prompt?: number;
  completion?: number;
  total?: number;
  [key: string]: any;
}

/**
 * Represents the function called by a tool, including its name and arguments.
 */
export interface ToolCallFunction {
  name?: string;
  arguments?: string | Record<string, any>;
  [key: string]: any;
}

/**
 * Represents a tool call, including its ID and the function invoked.
 */
export interface ToolCall {
  id?: string;
  function?: ToolCallFunction;
  [key: string]: any;
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
  [key: string]: any;
}

/**
 * Represents a text message content block.
 */
export interface TextMessageContent {
  type: "text";
  text: string;
  [key: string]: any;
}

/**
 * Represents an image message content block.
 */
export interface ImageMessageContent {
  type: "image";
  image: Image;
  [key: string]: any;
}

/**
 * Represents a tool with a JSON schema definition.
 */
export interface Tool {
  json_schema: string | Record<string, any>;
  [key: string]: any;
}
