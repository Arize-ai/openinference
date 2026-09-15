import type {
  InvokeModelCommand,
  InvokeModelResponse,
  InvokeModelWithResponseStreamCommand,
} from "@aws-sdk/client-bedrock-runtime";
import { diag } from "@opentelemetry/api";

import { isObjectWithStringKeys, withSafety } from "@arizeai/openinference-core";
import { LLMSystem } from "@arizeai/openinference-semantic-conventions";

import type {
  BedrockMessage,
  ExtendedConversationRole,
  ImageContent,
  ImageSource,
  InvokeModelRequestBody,
  MessageContent,
  TextContent,
  ToolResultContent,
  ToolUseContent,
  UsageAttributes,
} from "../types/bedrock-types";
import {
  isImageContent,
  isTextContent,
  isToolResultContent,
  isToolUseContent,
} from "../types/bedrock-types";

const isExtendedConversationRole = (value: unknown): value is ExtendedConversationRole =>
  value === "assistant" || value === "user" || value === "system" || value === "tool";

const isMessageContentBlock = (
  item: unknown,
): item is TextContent | ImageContent | ToolUseContent | ToolResultContent =>
  isTextContent(item) ||
  isImageContent(item) ||
  isToolUseContent(item) ||
  isToolResultContent(item);

/**
 * Coerces an unknown value into a {@link BedrockMessage} for telemetry capture, or returns
 * undefined when the value is not message-shaped. Unrecognized content blocks (e.g. thinking,
 * document) are dropped per-block so a single unknown block never erases the whole message
 * from the recorded input messages.
 */
const toBedrockMessage = (value: unknown): BedrockMessage | undefined => {
  if (!isObjectWithStringKeys(value) || !isExtendedConversationRole(value.role)) {
    return undefined;
  }
  if (typeof value.content === "string") {
    return { role: value.role, content: value.content };
  }
  if (Array.isArray(value.content)) {
    return { role: value.role, content: value.content.filter(isMessageContentBlock) };
  }
  return undefined;
};

/**
 * Type guard to check if message contains a simple single text content
 * Combines all checks needed to safely access the text content without casting
 *
 * @param message The bedrock message to check
 * @returns {boolean} True if message contains a single text content block
 */
export function isSimpleTextResponse(message: BedrockMessage): message is BedrockMessage & {
  content: [TextContent];
} {
  return Boolean(
    Array.isArray(message.content) &&
    message.content.length === 1 &&
    isTextContent(message.content[0]),
  );
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

// Request Processing Helpers

/**
 * Safely parses the InvokeModel request body with comprehensive error handling
 * Handles multiple body formats (string, Buffer, Uint8Array, ArrayBuffer) and provides fallback
 *
 * @param command The InvokeModelCommand containing the request body to parse
 * @returns {InvokeModelRequestBody | null} Parsed request body or null on error
 */
export const parseRequestBody = withSafety({
  fn: (
    command: InvokeModelCommand | InvokeModelWithResponseStreamCommand,
  ): InvokeModelRequestBody => {
    if (!command.input?.body) {
      throw new Error("Request body is missing");
    }

    let bodyString: string;
    if (typeof command.input.body === "string") {
      bodyString = command.input.body;
    } else if (Buffer.isBuffer(command.input.body)) {
      bodyString = command.input.body.toString("utf8");
    } else if (command.input.body instanceof Uint8Array) {
      bodyString = new TextDecoder().decode(command.input.body);
    } else if (command.input.body instanceof ArrayBuffer) {
      bodyString = new TextDecoder().decode(new Uint8Array(command.input.body));
    } else {
      throw new TypeError("Unsupported InvokeModel request body type");
    }
    const parsed: unknown = JSON.parse(bodyString);
    if (!isObjectWithStringKeys(parsed)) {
      throw new TypeError("InvokeModel request body must be a JSON object");
    }
    return parsed;
  },
  onError: (error) => {
    diag.warn("Error parsing InvokeModel request body:", error);
    return null;
  },
});

/**
 * Extracts invocation parameters from request body using AWS SDK standards
 * Maps snake_case parameter names to camelCase AWS SDK convention where applicable
 * Combines standard AWS SDK InferenceConfiguration with vendor-specific parameters
 *
 * @param requestBody The parsed request body containing model parameters
 * @param system The LLM system type to determine parameter extraction strategy
 * @returns {Record<string, unknown>} Object containing extracted invocation parameters
 */
export function extractInvocationParameters(
  requestBody: InvokeModelRequestBody,
  system: LLMSystem,
): Record<string, unknown> {
  if (system === LLMSystem.AMAZON && isObjectWithStringKeys(requestBody.inferenceConfig)) {
    return requestBody.inferenceConfig;
  } else if (
    system === LLMSystem.AMAZON &&
    isObjectWithStringKeys(requestBody.textGenerationConfig)
  ) {
    return requestBody.textGenerationConfig;
  } else {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { system, messages, tools, prompt, ...invocationParams } = requestBody;
    return invocationParams;
  }
}

/**
 * Extracts tool result content blocks from Bedrock message content
 * Filters content array to return only tool result blocks for processing tool responses
 *
 * @param content The message content to extract tool result blocks from
 * @returns {ToolResultContent[]} Array of tool result blocks, empty if none found
 */
export function extractToolResultBlocks(content: MessageContent): ToolResultContent[] {
  if (typeof content === "string" || !Array.isArray(content)) {
    return [];
  }

  return content.filter(isToolResultContent);
}

/**
 * Type guard to detect Amazon Nova request format
 * Checks for the characteristic structure: { messages: [{ role, content: [...] }] }
 *
 * @param requestBody The request body to check
 * @returns {boolean} True if request matches Nova format structure
 */
function isNovaRequest(requestBody: Record<string, unknown>): boolean {
  return (
    "messages" in requestBody &&
    Array.isArray(requestBody.messages) &&
    requestBody.messages.length > 0 &&
    typeof requestBody.messages[0] === "object" &&
    requestBody.messages[0] !== null &&
    "role" in requestBody.messages[0] &&
    "content" in requestBody.messages[0] &&
    Array.isArray(requestBody.messages[0].content)
  );
}

/**
 * Type guard to detect Amazon Titan request format
 * Checks for the characteristic structure: { inputText: string }
 *
 * @param requestBody The request body to check
 * @returns {boolean} True if request matches Titan format structure
 */
function isTitanRequest(requestBody: Record<string, unknown>): boolean {
  return hasStringProperty({ requestBody, key: "inputText" });
}

/**
 * Converts simple text-based request formats to standardized BedrockMessage array
 * Generic helper for models that use a single text field (like Titan's inputText or Cohere's prompt)
 * Creates a single user message with the provided text content
 *
 * @param requestBody The request body containing the text field
 * @param textFieldName The name of the field containing the text (e.g., 'inputText', 'prompt')
 * @returns {BedrockMessage[]} Single-element array containing the user message
 */
function convertSimpleTextToBedrockMessages(
  requestBody: Record<string, unknown>,
  textFieldName: string,
): BedrockMessage[] {
  const text = requestBody[textFieldName];
  if (typeof text !== "string") return [];

  return [
    {
      role: "user",
      content: text,
    },
  ];
}

/**
 * Converts Amazon Nova request format to standardized BedrockMessage array
 * Handles multi-modal content including text and images (video content is ignored)
 * Transforms Nova's nested content structure to flat BedrockMessage format
 *
 * @param requestBody The Nova-formatted request body containing messages
 * @returns {BedrockMessage[]} Array of normalized Bedrock messages with converted content
 */
function convertNovaToBedrockMessages(requestBody: Record<string, unknown>): BedrockMessage[] {
  const messages = Array.isArray(requestBody.messages)
    ? requestBody.messages.filter(isObjectWithStringKeys)
    : [];

  return messages.map((message): BedrockMessage => {
    const content: (TextContent | ImageContent)[] = [];

    const contentItems = Array.isArray(message.content)
      ? message.content.filter(isObjectWithStringKeys)
      : [];
    contentItems.forEach((contentItem) => {
      if (typeof contentItem.text === "string") {
        // Handle text content
        content.push({
          type: "text",
          text: contentItem.text,
        });
      } else if (isObjectWithStringKeys(contentItem.image)) {
        // Handle image content - always base64 string for Invoke API
        const imageData = contentItem.image;
        const source = isObjectWithStringKeys(imageData.source) ? imageData.source : undefined;
        if (typeof imageData.format !== "string" || typeof source?.bytes !== "string") return;

        content.push({
          type: "image",
          source: {
            type: "base64",
            media_type: `image/${imageData.format}`,
            data: source.bytes,
          },
        });
      }
      // Ignoring video content as requested
    });

    return {
      role: isExtendedConversationRole(message.role) ? message.role : "user",
      content,
    };
  });
}

/**
 * Type guard to detect Mistral Text Completion request format
 * Checks for the characteristic structure: { prompt: string }
 *
 * @param requestBody The request body to check
 * @returns {boolean} True if request matches Mistral Text Completion format structure
 */
function isMistralTextCompletionRequest(requestBody: Record<string, unknown>): boolean {
  return hasStringProperty({ requestBody, key: "prompt" });
}

/**
 * Type guard to detect Mistral Chat Completion request format (including Pixtral)
 * Checks for the characteristic structure: { messages: Array }
 *
 * @param requestBody The request body to check
 * @returns {boolean} True if request matches Mistral Chat format structure
 */
function isMistralChatRequest(requestBody: Record<string, unknown>): boolean {
  return (
    "messages" in requestBody &&
    Array.isArray(requestBody.messages) &&
    requestBody.messages.length > 0
  );
}

/**
 * Builds a single-text-content BedrockMessage from a Mistral message.
 *
 * @param params.message The Mistral message
 * @param params.role The already-normalized conversation role
 * @returns {BedrockMessage} The converted message
 */
function convertMistralTextMessage({
  message,
  role,
}: {
  message: Record<string, unknown>;
  role: ExtendedConversationRole;
}): BedrockMessage {
  return {
    role,
    content: [
      {
        type: "text",
        text: typeof message.content === "string" ? message.content : "",
      },
    ],
  };
}

/**
 * Converts a Mistral assistant message carrying `tool_calls` into a BedrockMessage.
 *
 * @param params.message The Mistral assistant message
 * @param params.role The already-normalized conversation role
 * @returns {BedrockMessage} The converted message
 */
function convertMistralAssistantToolCallsMessage({
  message,
  role,
}: {
  message: Record<string, unknown>;
  role: ExtendedConversationRole;
}): BedrockMessage {
  const rawToolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  const content: (TextContent | ToolUseContent)[] = [];
  for (const rawToolCall of rawToolCalls) {
    if (!isObjectWithStringKeys(rawToolCall)) continue;
    const fn = isObjectWithStringKeys(rawToolCall.function) ? rawToolCall.function : undefined;
    if (typeof fn?.name !== "string" || typeof fn.arguments !== "string") {
      continue;
    }
    const parsedInput: unknown = JSON.parse(fn.arguments);
    content.push({
      type: "tool_use",
      id: typeof rawToolCall.id === "string" ? rawToolCall.id : "unknown",
      name: fn.name,
      input: parsedInput,
    });
  }

  // Add text content if present
  if (message.content && typeof message.content === "string") {
    content.unshift({
      type: "text",
      text: message.content,
    });
  }

  return {
    role,
    content,
  };
}

/**
 * Converts a single Pixtral Large multimodal content block into BedrockMessage content.
 *
 * @param contentBlock The Mistral content block
 * @returns The converted content, or undefined for unsupported blocks
 */
function convertMistralMultimodalContentBlock(
  contentBlock: Record<string, unknown>,
): TextContent | ImageContent | undefined {
  if (contentBlock.type === "text" && typeof contentBlock.text === "string") {
    return {
      type: "text",
      text: contentBlock.text,
    };
  }
  if (
    contentBlock.type !== "image_url" ||
    !isObjectWithStringKeys(contentBlock.image_url) ||
    typeof contentBlock.image_url.url !== "string"
  ) {
    return undefined;
  }
  // Extract base64 data from data URL
  const base64Match = contentBlock.image_url.url.match(/^data:image\/([^;]+);base64,(.+)$/);
  if (!base64Match) {
    return undefined;
  }
  const [, format, base64Data] = base64Match;
  return {
    type: "image",
    source: {
      type: "base64",
      media_type: `image/${format}`,
      data: base64Data,
    },
  };
}

/**
 * Converts a Mistral message whose content is an array (Pixtral Large multimodal format).
 *
 * @param params.message The Mistral message
 * @param params.role The already-normalized conversation role
 * @returns {BedrockMessage} The converted message
 */
function convertMistralMultimodalMessage({
  message,
  role,
}: {
  message: Record<string, unknown>;
  role: ExtendedConversationRole;
}): BedrockMessage {
  const contentBlocks = Array.isArray(message.content) ? message.content : [];
  const content: (TextContent | ImageContent)[] = [];

  for (const contentBlock of contentBlocks.filter(isObjectWithStringKeys)) {
    const converted = convertMistralMultimodalContentBlock(contentBlock);
    if (converted != null) {
      content.push(converted);
    }
  }

  return {
    role,
    content,
  };
}

/**
 * Converts Mistral Chat Completion format to standardized BedrockMessage array
 * Handles complex message structures including tool calls and tool responses
 * Supports both regular chat and Pixtral Large (multimodal) formats
 *
 * @param requestBody The Mistral-formatted request body containing messages array
 * @returns {BedrockMessage[]} Array of converted BedrockMessage objects
 */
function convertMistralChatToBedrockMessages(
  requestBody: Record<string, unknown>,
): BedrockMessage[] {
  const messages = Array.isArray(requestBody.messages)
    ? requestBody.messages.filter(isObjectWithStringKeys)
    : [];

  return messages.map((message): BedrockMessage => {
    const role = isExtendedConversationRole(message.role) ? message.role : "user";
    // Handle tool role messages (Mistral-specific)
    if (role === "tool") {
      return convertMistralTextMessage({ message, role });
    }

    // Handle assistant messages with tool calls
    if (role === "assistant" && Array.isArray(message.tool_calls)) {
      return convertMistralAssistantToolCallsMessage({ message, role });
    }

    // Handle Pixtral Large multimodal content (array format)
    if (Array.isArray(message.content)) {
      return convertMistralMultimodalMessage({ message, role });
    }

    // Handle regular text content (string format)
    // Edge case: Simple text messages mixed in with complex chat completion requests
    return convertMistralTextMessage({ message, role });
  });
}

/**
 * Converts AI21 Jamba request format to standardized BedrockMessage array
 * Handles messages with system, user, and assistant roles
 * Similar to Anthropic format but supports system role like Mistral
 *
 * @param requestBody The AI21 Jamba-formatted request body containing messages array
 * @returns {BedrockMessage[]} Array of converted BedrockMessage objects
 */
function convertAI21JambaToBedrockMessages(requestBody: Record<string, unknown>): BedrockMessage[] {
  const messages = Array.isArray(requestBody.messages)
    ? requestBody.messages.filter(isObjectWithStringKeys)
    : [];

  return messages.map(
    (message): BedrockMessage => ({
      role: isExtendedConversationRole(message.role) ? message.role : "user",
      content: [
        {
          type: "text",
          text: typeof message.content === "string" ? message.content : "",
        },
      ],
    }),
  );
}

/**
 * Fallback normalization for unknown model request formats
 * Attempts pattern-based detection when LLM system identification fails
 *
 * This function provides graceful degradation for cases where the model ID
 * doesn't map to a known LLM system, using common request structure patterns
 * to make a best-effort conversion to BedrockMessage format.
 *
 * @param requestBody The raw request body from an unknown model format
 * @returns {BedrockMessage[]} Array of normalized messages, or empty array if no patterns match
 *
 * @internal Used as last resort when LLM system detection fails
 *
 * @example
 * // Handles messages-based format (Anthropic-like)
 * fallbackNormalizeRequestContentBlocks({ messages: [{ role: "user", content: "hi" }] })
 *
 * @example
 * // Handles prompt-based format (completion models)
 * fallbackNormalizeRequestContentBlocks({ prompt: "Hello world" })
 */
function fallbackNormalizeRequestContentBlocks(
  requestBody: Record<string, unknown>,
): BedrockMessage[] {
  if (
    "messages" in requestBody &&
    Array.isArray(requestBody.messages) &&
    requestBody.messages.length > 0
  ) {
    return requestBody.messages.flatMap((message) => toBedrockMessage(message) ?? []);
  } else if ("prompt" in requestBody && typeof requestBody.prompt === "string") {
    return convertSimpleTextToBedrockMessages(requestBody, "prompt");
  } else if ("inputText" in requestBody && typeof requestBody.inputText === "string") {
    return convertSimpleTextToBedrockMessages(requestBody, "inputText");
  }
  return [];
}

/**
 * Returns true when the request body carries a string value at the given key.
 */
function hasStringProperty({
  requestBody,
  key,
}: {
  requestBody: Record<string, unknown>;
  key: "prompt" | "inputText";
}): boolean {
  return key in requestBody && typeof requestBody[key] === "string";
}

/**
 * Normalizes Amazon request bodies (Nova multi-modal messages or Titan simple text).
 *
 * @param requestBody The parsed Amazon request body
 * @returns {BedrockMessage[]} Array of normalized Bedrock messages
 */
function normalizeAmazonRequestContentBlocks(
  requestBody: InvokeModelRequestBody,
): BedrockMessage[] {
  if (isNovaRequest(requestBody)) {
    // Handle Amazon Nova format: { messages: [{ role, content: [{ text }] }] }
    return convertNovaToBedrockMessages(requestBody);
  }
  if (isTitanRequest(requestBody)) {
    // vs Titan format: { inputText: string }
    return convertSimpleTextToBedrockMessages(requestBody, "inputText");
  }
  // LLM system defaults to Amazon when no correct format is given
  // In this case we should gracefully degrade and extract as much info as possible
  return fallbackNormalizeRequestContentBlocks(requestBody);
}

/**
 * Normalizes Mistral request bodies (Chat/Pixtral messages or text completion prompt).
 *
 * @param requestBody The parsed Mistral request body
 * @returns {BedrockMessage[]} Array of normalized Bedrock messages, empty for unknown shapes
 */
function normalizeMistralRequestContentBlocks(
  requestBody: InvokeModelRequestBody,
): BedrockMessage[] {
  if (isMistralChatRequest(requestBody)) {
    // Handle Mistral Chat/Pixtral format: { messages: [{ role, content }] }
    return convertMistralChatToBedrockMessages(requestBody);
  }
  if (isMistralTextCompletionRequest(requestBody)) {
    // Handle Mistral Text Completion format: { prompt: string }
    return convertSimpleTextToBedrockMessages(requestBody, "prompt");
  }
  return [];
}

/**
 * Normalizes AI21 request bodies (completion prompt or Jamba messages).
 *
 * @param requestBody The parsed AI21 request body
 * @returns {BedrockMessage[]} Array of normalized Bedrock messages
 */
function normalizeAI21RequestContentBlocks(requestBody: InvokeModelRequestBody): BedrockMessage[] {
  if (hasStringProperty({ requestBody, key: "prompt" })) {
    // Handle AI21 format: { prompt: string }
    return convertSimpleTextToBedrockMessages(requestBody, "prompt");
  }
  if (
    "messages" in requestBody &&
    Array.isArray(requestBody.messages) &&
    requestBody.messages.length > 0
  ) {
    // Handle AI21 Jamba format: { messages: Array }
    return convertAI21JambaToBedrockMessages(requestBody);
  }
  return fallbackNormalizeRequestContentBlocks(requestBody);
}

/**
 * Normalizes request content blocks from various model providers into standardized BedrockMessage format
 * Handles Amazon Nova (multi-modal messages), Titan (simple text), Anthropic, and other providers
 * Provides error handling and fallback to empty array on normalization failures
 *
 * @param requestBody The parsed request body containing messages in provider-specific format
 * @param llm_system The LLM system type to determine normalization strategy
 * @returns {BedrockMessage[]} Array of normalized Bedrock messages or empty array on error
 */
export const normalizeRequestContentBlocks = withSafety({
  fn: (requestBody: InvokeModelRequestBody, llm_system: LLMSystem): BedrockMessage[] => {
    switch (llm_system) {
      case LLMSystem.ANTHROPIC:
        return Array.isArray(requestBody.messages)
          ? requestBody.messages.flatMap((message) => toBedrockMessage(message) ?? [])
          : [];
      case LLMSystem.AMAZON:
        return normalizeAmazonRequestContentBlocks(requestBody);
      case LLMSystem.COHERE:
      case LLMSystem.META:
        // Handle Cohere and Meta formats: { prompt: string }
        return hasStringProperty({ requestBody, key: "prompt" })
          ? convertSimpleTextToBedrockMessages(requestBody, "prompt")
          : fallbackNormalizeRequestContentBlocks(requestBody);
      case LLMSystem.MISTRALAI:
        return normalizeMistralRequestContentBlocks(requestBody);
      case LLMSystem.AI21:
        return normalizeAI21RequestContentBlocks(requestBody);
      default:
        return fallbackNormalizeRequestContentBlocks(requestBody);
    }
  },
  onError: (error) => {
    diag.warn("Error normalizing request content blocks:", error);
    return [];
  },
});

// Response Processing Helpers

/**
 * Safely parses the InvokeModel response body with comprehensive error handling
 * Handles multiple response body formats and provides null fallback on error
 *
 * @param response The raw InvokeModel command response from AWS SDK
 * @returns {Record<string, unknown> | null} Parsed response body or null if parsing fails
 * @internal Used by response attribute extraction functions
 */
export const parseResponseBody = withSafety({
  fn: (response: InvokeModelResponse): Record<string, unknown> => {
    if (!response.body) {
      throw new Error("Response body is missing");
    }

    let responseText: string;
    if (typeof response.body === "string") {
      responseText = response.body;
    } else if (response.body instanceof Uint8Array) {
      responseText = new TextDecoder().decode(response.body);
    } else {
      throw new TypeError("Unsupported InvokeModel response body type");
    }

    const parsed: unknown = JSON.parse(responseText);
    if (!isObjectWithStringKeys(parsed)) {
      throw new TypeError("InvokeModel response body must be a JSON object");
    }
    return parsed;
  },
  onError: (error) => {
    diag.warn("Error parsing response body:", error);
    return null;
  },
});

/**
 * Coerces Nova-style content blocks to standard MessageContent format
 * Works with raw JSON structure without importing Nova types
 * Handles text content and tool use blocks with proper type transformation
 *
 * @param content The raw content array from Nova response to transform
 * @returns {MessageContent} Transformed content in standard Bedrock format
 */
export function coerceNovaToMessageContent(content: unknown): MessageContent {
  if (!Array.isArray(content)) {
    return [];
  }

  const transformedContent = content
    .map((block): TextContent | ToolUseContent | null => {
      if (!isObjectWithStringKeys(block)) {
        return null;
      }

      const obj = block;

      // Nova text content: { text: string } -> { type: "text", text: string }
      if ("text" in obj && typeof obj.text === "string" && !("type" in obj)) {
        return {
          type: "text",
          text: obj.text,
        };
      }

      // Nova tool use: { toolUse: { toolUseId, name, input } } -> { type: "tool_use", id, name, input }
      if (isObjectWithStringKeys(obj.toolUse)) {
        const toolUse = obj.toolUse;
        if (
          typeof toolUse.toolUseId === "string" &&
          typeof toolUse.name === "string" &&
          isObjectWithStringKeys(toolUse.input)
        ) {
          return {
            type: "tool_use",
            id: toolUse.toolUseId,
            name: toolUse.name,
            input: toolUse.input,
          };
        }
      }

      return null;
    })
    .filter((item): item is TextContent | ToolUseContent => item !== null);

  return transformedContent;
}

/**
 * Extracts Nova content from the nested response structure
 * Nova format: { output: { message: { content: [...] } } }
 *
 * @param responseBody The raw response body from Nova model
 * @returns {unknown} Extracted content array or empty array if structure is invalid
 */
function extractNovaContent(responseBody: Record<string, unknown>): unknown {
  const output = isObjectWithStringKeys(responseBody.output) ? responseBody.output : undefined;
  if (output == null) return [];

  const message = isObjectWithStringKeys(output.message) ? output.message : undefined;
  if (message == null) return [];

  return message.content || [];
}

/**
 * Type guard to identify Nova response format
 * Checks for the characteristic nested structure: { output: { message: ... } }
 *
 * @param responseBody The response body to check
 * @returns {boolean} True if response matches Nova format structure
 */
function isNovaResponse(responseBody: Record<string, unknown>): boolean {
  return isObjectWithStringKeys(responseBody.output) && responseBody.output.message != null;
}

/**
 * Type guard to identify Titan response format
 * Checks for the characteristic structure: { results: [...], inputTextTokenCount: number }
 *
 * @param responseBody The response body to check
 * @returns {boolean} True if response matches Titan format structure
 */
function isTitanResponse(responseBody: Record<string, unknown>): boolean {
  return !!(
    responseBody.results &&
    Array.isArray(responseBody.results) &&
    responseBody.results.length > 0 &&
    typeof responseBody.inputTextTokenCount === "number"
  );
}

/**
 * Converts AI21 Jamba response to standardized MessageContent format
 * Handles the choices array structure: { choices: [{ message: { content, tool_calls } }] }
 * Supports both plain text responses and tool call responses
 *
 * @param responseBody The AI21 Jamba response body to convert
 * @returns {MessageContent} Array of converted content blocks including tool calls
 */
function convertAI21JambaToMessageContent(responseBody: Record<string, unknown>): MessageContent {
  if (!Array.isArray(responseBody.choices)) {
    return [];
  }

  const content: MessageContent = [];
  const choices = responseBody.choices;

  for (const choice of choices) {
    if (isObjectWithStringKeys(choice)) {
      const message = isObjectWithStringKeys(choice.message) ? choice.message : undefined;

      if (message) {
        if (typeof message.content === "string") {
          content.push({
            type: "text",
            text: message.content,
          });
        }

        // Handle tool calls - AI21 format: { tool_calls: [{ id, function: { name, arguments } }] }
        if (Array.isArray(message.tool_calls)) {
          for (const toolCall of message.tool_calls.filter(isObjectWithStringKeys)) {
            const fn = isObjectWithStringKeys(toolCall.function) ? toolCall.function : undefined;
            if (typeof fn?.name === "string" && typeof fn.arguments === "string") {
              try {
                content.push({
                  type: "tool_use",
                  id: typeof toolCall.id === "string" ? toolCall.id : "unknown",
                  name: fn.name,
                  input: JSON.parse(fn.arguments),
                });
              } catch (error) {
                // If arguments parsing fails, skip this tool call
                diag.warn("Failed to parse AI21 tool call arguments:", error);
              }
            }
          }
        }
      }
    }
  }

  return content;
}

/**
 * Converts Meta response to standardized MessageContent format
 * Handles the single generation field: { generation: "text" }
 *
 * @param responseBody The Meta response body to convert
 * @returns {MessageContent} Array with single converted content block
 */
export function convertMetaToMessageContent(responseBody: Record<string, unknown>): MessageContent {
  const generation = responseBody.generation;
  if (typeof generation === "string") {
    return [
      {
        type: "text",
        text: generation,
      },
    ];
  }
  return [];
}

/**
 * Converts an array field in response body to MessageContent array
 * Handles multiple generations/results by converting each element to a TextContent block
 * Provides validation to ensure the field exists and is an array
 *
 * @param responseBody The parsed response body containing the array field
 * @param arrayFieldName The name of the array field in the response body (e.g., "generations", "results")
 * @param textFieldName The name of the text field within each array element (e.g., "text", "outputText")
 * @returns {MessageContent} Array of TextContent blocks, one for each element in the source array
 */
function convertArrayFieldToMessageContent(
  responseBody: Record<string, unknown>,
  arrayFieldName: string,
  textFieldName: string,
): MessageContent {
  // Validate that the array field exists and is actually an array
  const arrayField = responseBody[arrayFieldName];
  if (!Array.isArray(arrayField) || arrayField.length === 0) {
    return [];
  }

  // Convert each element in the array to a TextContent block
  const content: TextContent[] = [];
  for (const element of arrayField) {
    if (isObjectWithStringKeys(element)) {
      const text = element[textFieldName];
      if (typeof text === "string") {
        content.push({
          type: "text",
          text: text,
        });
      }
    }
  }

  return content;
}

/**
 * Normalizes Amazon response bodies, distinguishing Nova from Titan by response structure.
 *
 * @param responseBody The parsed Amazon response body
 * @returns {MessageContent} The extracted message content, empty for unknown shapes
 */
function normalizeAmazonResponseContent(responseBody: Record<string, unknown>): MessageContent {
  if (isNovaResponse(responseBody)) {
    return coerceNovaToMessageContent(extractNovaContent(responseBody));
  }
  if (isTitanResponse(responseBody)) {
    // Titan format: { results: [{ outputText }] } - handle all results, not just first
    return convertArrayFieldToMessageContent(responseBody, "results", "outputText");
  }
  return [];
}

/**
 * Extracts the assistant message content from a provider-specific response body.
 *
 * @param responseBody The parsed response body containing content in provider-specific format
 * @param llm_system The LLM system type to determine normalization strategy
 * @returns {MessageContent} The extracted message content, empty for unknown shapes
 */
function normalizeResponseContent({
  responseBody,
  llm_system,
}: {
  responseBody: Record<string, unknown>;
  llm_system: LLMSystem;
}): MessageContent {
  switch (llm_system) {
    case LLMSystem.ANTHROPIC:
      // Anthropic format: { content: [{ type: "text", text: "..." }] }
      return Array.isArray(responseBody.content)
        ? responseBody.content.filter(isMessageContentBlock)
        : [];
    case LLMSystem.AMAZON:
      return normalizeAmazonResponseContent(responseBody);
    case LLMSystem.COHERE:
    case LLMSystem.MISTRALAI:
      // Cohere and Mistral: { generations: [{ text }] } - handle all generations, not just first
      // NOTE: Tool calls are not currently supported for Mistral models
      return convertArrayFieldToMessageContent(responseBody, "generations", "text");
    case LLMSystem.META:
      return typeof responseBody.generation === "string"
        ? convertMetaToMessageContent(responseBody)
        : [];
    case LLMSystem.AI21:
      return convertAI21JambaToMessageContent(responseBody);
    default:
      return [];
  }
}

/**
 * Normalizes response content blocks from various model providers into standardized BedrockMessage format
 * Handles Amazon Nova (nested output structure), Titan (results array), Anthropic, and other providers
 * Provides error handling and fallback to empty assistant message on normalization failures
 *
 * @param responseBody The parsed response body containing content in provider-specific format
 * @param llm_system The LLM system type to determine normalization strategy
 * @returns {BedrockMessage} Normalized assistant message with extracted content or empty fallback
 */
export const normalizeResponseContentBlocks = withSafety({
  fn: (responseBody: Record<string, unknown>, llm_system: LLMSystem): BedrockMessage => ({
    role: "assistant",
    content: normalizeResponseContent({ responseBody, llm_system }),
  }),
  onError: (error) => {
    diag.warn("Error normalizing content blocks:", error);
    return {
      role: "assistant",
      content: [],
    };
  },
});

/**
 * Returns the value at `key` when it is a number, otherwise undefined.
 */
function getNumberProperty({
  source,
  key,
}: {
  source: Record<string, unknown>;
  key: string;
}): number | undefined {
  const value = source[key];
  return typeof value === "number" ? value : undefined;
}

/**
 * Returns the response body's `usage` object when present.
 */
function getUsageObject(
  responseBody: Record<string, unknown>,
): Record<string, unknown> | undefined {
  return isObjectWithStringKeys(responseBody.usage) ? responseBody.usage : undefined;
}

/**
 * Normalizes Anthropic usage.
 * Format: `{ usage: { input_tokens, output_tokens, total_tokens?, cache_read_input_tokens?, cache_creation_input_tokens? } }`
 */
function normalizeAnthropicUsage(responseBody: Record<string, unknown>): UsageAttributes {
  const usage = getUsageObject(responseBody);
  if (!usage) return {};

  return {
    input_tokens: getNumberProperty({ source: usage, key: "input_tokens" }),
    output_tokens: getNumberProperty({ source: usage, key: "output_tokens" }),
    total_tokens: getNumberProperty({ source: usage, key: "total_tokens" }),
    cache_read_input_tokens: getNumberProperty({ source: usage, key: "cache_read_input_tokens" }),
    cache_creation_input_tokens: getNumberProperty({
      source: usage,
      key: "cache_creation_input_tokens",
    }),
  };
}

/**
 * Normalizes Amazon Nova usage.
 * Format: `{ usage: { inputTokens, outputTokens, totalTokens?, cacheReadInputTokenCount?, cacheWriteInputTokenCount? } }`
 */
function normalizeNovaUsage(responseBody: Record<string, unknown>): UsageAttributes {
  const usage = getUsageObject(responseBody);
  if (!usage) return {};

  return {
    input_tokens: getNumberProperty({ source: usage, key: "inputTokens" }),
    output_tokens: getNumberProperty({ source: usage, key: "outputTokens" }),
    total_tokens: getNumberProperty({ source: usage, key: "totalTokens" }),
    cache_read_input_tokens: getNumberProperty({ source: usage, key: "cacheReadInputTokenCount" }),
    cache_creation_input_tokens: getNumberProperty({
      source: usage,
      key: "cacheWriteInputTokenCount",
    }),
  };
}

/**
 * Normalizes Amazon Titan usage.
 * Format: `{ inputTextTokenCount: N, results: [{ tokenCount: N }] }`
 */
function normalizeTitanUsage(responseBody: Record<string, unknown>): UsageAttributes {
  const inputTokens = getNumberProperty({ source: responseBody, key: "inputTextTokenCount" });
  const results = Array.isArray(responseBody.results)
    ? responseBody.results.filter(isObjectWithStringKeys)
    : [];
  const firstResult = results[0];
  const outputTokens =
    firstResult != null ? getNumberProperty({ source: firstResult, key: "tokenCount" }) : undefined;

  const result: UsageAttributes = {};
  if (inputTokens !== undefined) result.input_tokens = inputTokens;
  if (outputTokens !== undefined) result.output_tokens = outputTokens;
  return result;
}

/**
 * Normalizes Amazon usage, which differs between Nova and Titan responses.
 */
function normalizeAmazonUsage(responseBody: Record<string, unknown>): UsageAttributes {
  if (isNovaResponse(responseBody)) {
    return normalizeNovaUsage(responseBody);
  }
  if (isTitanResponse(responseBody)) {
    return normalizeTitanUsage(responseBody);
  }
  return {};
}

/**
 * Normalizes AI21 Jamba usage.
 * Format: `{ usage: { prompt_tokens, completion_tokens, total_tokens } }`
 */
function normalizeAI21Usage(responseBody: Record<string, unknown>): UsageAttributes {
  const usage = getUsageObject(responseBody);
  if (!usage) return {};

  return {
    input_tokens: getNumberProperty({ source: usage, key: "prompt_tokens" }),
    output_tokens: getNumberProperty({ source: usage, key: "completion_tokens" }),
    total_tokens: getNumberProperty({ source: usage, key: "total_tokens" }),
  };
}

/**
 * Normalizes token usage information from various model providers into standardized format
 * Handles different provider token count field names and structures including caching tokens
 * Returns comprehensive UsageAttributes with all available token information
 *
 * @param responseBody The parsed response body containing usage information
 * @param llm_system The LLM system type to determine extraction strategy
 * @returns {UsageAttributes} Normalized usage object with comprehensive token information
 */
export const normalizeUsageAttributes = withSafety({
  fn: (responseBody: Record<string, unknown>, llm_system: LLMSystem): UsageAttributes => {
    switch (llm_system) {
      case LLMSystem.ANTHROPIC:
        return normalizeAnthropicUsage(responseBody);
      case LLMSystem.AMAZON:
        return normalizeAmazonUsage(responseBody);
      case LLMSystem.AI21:
        return normalizeAI21Usage(responseBody);
      case LLMSystem.META:
        // Meta format: { prompt_token_count: N, generation_token_count: N }
        return {
          input_tokens: getNumberProperty({ source: responseBody, key: "prompt_token_count" }),
          output_tokens: getNumberProperty({ source: responseBody, key: "generation_token_count" }),
        };
      // Cohere reports token counts in HTTP headers rather than the response body, and
      // Mistral reports none at all, so both fall through to the empty default below.
      default:
        return {};
    }
  },
  onError: (error) => {
    diag.warn("Error normalizing usage attributes:", error);
    return {};
  },
});
