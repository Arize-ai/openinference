import { withSafety } from "@arizeai/openinference-core";
import { InvokeModelCommand, InvokeModelResponse } from "@aws-sdk/client-bedrock-runtime";
import { diag } from "@opentelemetry/api";
import {
  BedrockMessage,
  InvokeModelRequestBody,
  isImageContent,
  isTextContent,
  isToolResultContent,
  isToolUseContent,
  MessageContent,
  TextContent,
  ToolUseContent,
  ImageSource,
  ToolResultContent,
  ImageContent,
  ConversationRole,
  ExtendedConversationRole,
} from "../types/bedrock-types";
import { LLMSystem } from "@arizeai/openinference-semantic-conventions";

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
      // Handle other potential types
      responseText = new TextDecoder().decode(response.body as Uint8Array);
    }

    return JSON.parse(responseText) as Record<string, unknown>;
  },
  onError: (error) => {
    diag.warn("Error parsing response body:", error);
    return null;
  },
});

/**
 * Safely parses the InvokeModel request body with comprehensive error handling
 * Handles multiple body formats (string, Buffer, Uint8Array, ArrayBuffer) and provides fallback
 *
 * @param command The InvokeModelCommand containing the request body to parse
 * @returns {InvokeModelRequestBody | null} Parsed request body or null on error
 */
export const parseRequestBody = withSafety({
    fn: (command: InvokeModelCommand): InvokeModelRequestBody => {
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
        // For other types, convert to string safely
        bodyString = String(command.input.body);
      }
      return JSON.parse(bodyString) as InvokeModelRequestBody;
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
    if (system === LLMSystem.AMAZON && requestBody.inferenceConfig && 
        typeof requestBody.inferenceConfig === 'object' && requestBody.inferenceConfig !== null) {
      return requestBody.inferenceConfig as Record<string, unknown>;
    } else if (system === LLMSystem.AMAZON && requestBody.textGenerationConfig && 
               typeof requestBody.textGenerationConfig === 'object' && requestBody.textGenerationConfig !== null) {
      return requestBody.textGenerationConfig as Record<string, unknown>;
    } else {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const {system, messages, tools, prompt, ...invocationParams} = requestBody;
      return invocationParams;
    }
  }

/**
 * Type guard to check if message contains a simple single text content
 * Combines all checks needed to safely access the text content without casting
 *
 * @param message The bedrock message to check
 * @returns {boolean} True if message contains a single text content block
 */
export function isSimpleTextResponse(
  message: BedrockMessage,
): message is BedrockMessage & {
  content: [TextContent];
} {
  return Boolean(
    Array.isArray(message.content) &&
    message.content.length === 1 &&
    isTextContent(message.content[0])
  );
}


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
 * @deprecated This function is currently unused but preserved for potential future use
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
 * @deprecated This function is currently unused but preserved for potential future use
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
 * @deprecated This function is currently unused but preserved for potential future use
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


/**
 * Type guard to detect Amazon Nova request format
 * Checks for the characteristic structure: { messages: [{ role, content: [...] }] }
 *
 * @param requestBody The request body to check
 * @returns {boolean} True if request matches Nova format structure
 */
function isNovaRequest(requestBody: Record<string, unknown>): boolean {
    return (
      'messages' in requestBody &&
      Array.isArray(requestBody.messages) &&
      requestBody.messages.length > 0 &&
      typeof requestBody.messages[0] === 'object' &&
      requestBody.messages[0] !== null &&
      'role' in requestBody.messages[0] &&
      'content' in requestBody.messages[0] &&
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
    return 'inputText' in requestBody && typeof requestBody.inputText === 'string';
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
  textFieldName: string
): BedrockMessage[] {
  const text = requestBody[textFieldName] as string;
  
  return [
    {
      role: "user" as ConversationRole,
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
  const messages = requestBody.messages as Array<{
    role: string;
    content: Array<{
      text?: string;
      image?: {
        format: string; // Always base64 string for Invoke API
        source: {
          bytes: string; // Always base64 string for Invoke API
        };
      };
      video?: unknown; // Ignoring video for now
    }>;
  }>;

  return messages.map((message) => {
    const content: (TextContent | ImageContent)[] = [];

    

    message.content.forEach((contentItem) => {
      if (contentItem.text) {
        // Handle text content
        content.push({
          type: "text",
          text: contentItem.text,
        });
      } else if (contentItem.image) {
        // Handle image content - always base64 string for Invoke API
        const imageData = contentItem.image;
        const mimeType = `image/${imageData.format}`;

        content.push({
          type: "image",
          source: {
            type: "base64",
            media_type: mimeType,
            data: imageData.source.bytes, // Already base64 string
          },
        });
      }
      // Ignoring video content as requested
    });

    return {
      role: message.role as ConversationRole,
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
  return 'prompt' in requestBody && typeof requestBody.prompt === 'string';
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
    'messages' in requestBody &&
    Array.isArray(requestBody.messages) &&
    requestBody.messages.length > 0
  );
}

/**
 * Converts Mistral Chat Completion format to standardized BedrockMessage array
 * Handles complex message structures including tool calls and tool responses
 * Supports both regular chat and Pixtral Large (multimodal) formats
 *
 * @param requestBody The Mistral-formatted request body containing messages array
 * @returns {BedrockMessage[]} Array of converted BedrockMessage objects
 */
function convertMistralChatToBedrockMessages(requestBody: Record<string, unknown>): BedrockMessage[] {
  const messages = requestBody.messages as Array<{
    role: string;
    content?: string | Array<{
      type?: string;
      text?: string;
      image_url?: {
        url: string;
      };
    }>;
    tool_calls?: Array<{
      id: string;
      function: {
        name: string;
        arguments: string;
      };
    }>;
    tool_call_id?: string;
  }>;

  return messages.map((message) => {
    // Handle tool role messages (Mistral-specific)
    if (message.role === "tool") {
      return {
        role: "tool" as ExtendedConversationRole,
        content: [
          {
            type: "text",
            text: typeof message.content === "string" ? message.content : "",
          },
        ],
      };
    }

    // Handle assistant messages with tool calls
    if (message.role === "assistant" && message.tool_calls) {
      const content: (TextContent | ToolUseContent)[] = message.tool_calls.map((toolCall) => ({
        type: "tool_use",
        id: toolCall.id,
        name: toolCall.function.name,
        input: JSON.parse(toolCall.function.arguments),
      }));

      // Add text content if present
      if (message.content && typeof message.content === "string") {
        content.unshift({
          type: "text",
          text: message.content,
        });
      }
      
      // Edge case: Simple text messages mixed in with complex chat completion requests
      return {
        role: message.role as ExtendedConversationRole,
        content,
      };
    }

    // Handle Pixtral Large multimodal content (array format)
    if (Array.isArray(message.content)) {
      const content: (TextContent | ImageContent)[] = [];

      for (const contentBlock of message.content) {
        if (contentBlock.type === "text" && contentBlock.text) {
          content.push({
            type: "text",
            text: contentBlock.text,
          });
        } else if (contentBlock.type === "image_url" && contentBlock.image_url?.url) {
          // Extract base64 data from data URL
          const dataUrl = contentBlock.image_url.url;
          const base64Match = dataUrl.match(/^data:image\/([^;]+);base64,(.+)$/);
          
          if (base64Match) {
            const [, format, base64Data] = base64Match;
            content.push({
              type: "image",
              source: {
                type: "base64",
                media_type: `image/${format}`,
                data: base64Data,
              },
            });
          }
        }
      }

      return {
        role: message.role as ExtendedConversationRole,
        content,
      };
    }

    // Handle regular text content (string format)
    // Edge case: Simple text messages mixed in with complex chat completion requests
    return {
      role: message.role as ExtendedConversationRole,
      content: [
        {
          type: "text",
          text: typeof message.content === "string" ? message.content : "",
        },
      ],
    };
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
    const messages = requestBody.messages as Array<{
      role: string;
      content: string;
    }>;
  
    return messages.map((message) => ({
      role: message.role as ExtendedConversationRole,
      content: [
        {
          type: "text",
          text: message.content,
        },
      ],
    }));
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
function fallbackNormalizeRequestContentBlocks(requestBody:  Record<string, unknown>): BedrockMessage[] {
    if ('messages' in requestBody && Array.isArray(requestBody.messages) && requestBody.messages.length > 0) {
        return requestBody.messages as BedrockMessage[];
    } else if ('prompt' in requestBody && typeof requestBody.prompt === 'string') {
        return  convertSimpleTextToBedrockMessages(requestBody, 'prompt');
    } else if ('inputText' in requestBody && typeof requestBody.inputText === 'string') {
        return convertSimpleTextToBedrockMessages(requestBody, 'inputText');
    }
    return [];
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
    fn: (
      requestBody: InvokeModelRequestBody,
      llm_system: LLMSystem,
    ): BedrockMessage[] => {
      let messages: BedrockMessage[] = [];

      if (llm_system === LLMSystem.ANTHROPIC) {
        messages = requestBody.messages as BedrockMessage[];
      } else if (llm_system === LLMSystem.AMAZON) {
        if (isNovaRequest(requestBody)) {
            // Handle Amazon Nova format: { messages: [{ role, content: [{ text }] }] }
            messages = convertNovaToBedrockMessages(requestBody);
        } else if (isTitanRequest(requestBody)) {
             // vs Titan format: { inputText: string }
            messages = convertSimpleTextToBedrockMessages(requestBody, 'inputText');
        } else {
            // LLM system defaults to Amazon when no correct format is given
            // In this case we should gracefully degrade and extract as much info as possible
            messages = fallbackNormalizeRequestContentBlocks(requestBody);
        }
      } else if (llm_system === LLMSystem.COHERE && 'prompt' in requestBody && typeof requestBody.prompt === 'string') {
        // Handle Cohere format: { prompt: string }
        messages = convertSimpleTextToBedrockMessages(requestBody, 'prompt');
      } else if (llm_system === LLMSystem.META && 'prompt' in requestBody && typeof requestBody.prompt === 'string') {
        // Handle Meta format: { prompt: string }
        messages = convertSimpleTextToBedrockMessages(requestBody, 'prompt');
      } else if (llm_system === LLMSystem.MISTRALAI) {
        // Handle Mistral formats
        if (isMistralChatRequest(requestBody)) {
          // Handle Mistral Chat/Pixtral format: { messages: [{ role, content }] }
          messages = convertMistralChatToBedrockMessages(requestBody);
        } else if (isMistralTextCompletionRequest(requestBody)) {
          // Handle Mistral Text Completion format: { prompt: string }
          messages = convertSimpleTextToBedrockMessages(requestBody, 'prompt');
        }
      } else if (llm_system === LLMSystem.AI21 && 'prompt' in requestBody && typeof requestBody.prompt === 'string') {
        // Handle AI21 format: { prompt: string }
        messages = convertSimpleTextToBedrockMessages(requestBody, 'prompt');
      } else if (llm_system === LLMSystem.AI21 && 'messages' in requestBody && Array.isArray(requestBody.messages) && requestBody.messages.length > 0) {
        // Handle AI21 Jamba format: { messages: Array }
        messages = convertAI21JambaToBedrockMessages(requestBody);
      } else {
        messages = fallbackNormalizeRequestContentBlocks(requestBody);
      }

      return messages;
    },
    onError: (error) => {
      diag.warn("Error normalizing request content blocks:", error);
      return [];
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
function coerceNovaToMessageContent(content: unknown): MessageContent {
  if (!Array.isArray(content)) {
    return [];
  }

  const transformedContent = content
    .map((block): TextContent | ToolUseContent | null => {
      if (!block || typeof block !== "object") {
        return null;
      }

      const obj = block as Record<string, unknown>;

      // Nova text content: { text: string } -> { type: "text", text: string }
      if ("text" in obj && typeof obj.text === "string" && !("type" in obj)) {
        return {
          type: "text",
          text: obj.text,
        };
      }

      // Nova tool use: { toolUse: { toolUseId, name, input } } -> { type: "tool_use", id, name, input }
      if (
        "toolUse" in obj &&
        typeof obj.toolUse === "object" &&
        obj.toolUse !== null
      ) {
        const toolUse = obj.toolUse as Record<string, unknown>;

        if ("toolUseId" in toolUse && "name" in toolUse && "input" in toolUse) {
          return {
            type: "tool_use",
            id: toolUse.toolUseId as string,
            name: toolUse.name as string,
            input: toolUse.input as Record<string, unknown>,
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
  const output = responseBody.output as Record<string, unknown> | undefined;
  if (!output) return [];

  const message = output.message as Record<string, unknown> | undefined;
  if (!message) return [];

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
  return !!(
    responseBody.output &&
    typeof responseBody.output === "object" &&
    (responseBody.output as Record<string, unknown>).message
  );
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
    typeof responseBody.inputTextTokenCount === "number"
  );
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
  fn: (
    responseBody: Record<string, unknown>,
    llm_system: LLMSystem,
  ): BedrockMessage => {
    const role = "assistant";
    let content: MessageContent = [];

    if (llm_system === LLMSystem.ANTHROPIC) {
      // Anthropic already in correct format: { content: [{ type: "text", text: "..." }] }
      content = responseBody.content as MessageContent;
    } else if (llm_system === LLMSystem.AMAZON) {
      // Distinguish between Nova and Titan by response structure
      if (isNovaResponse(responseBody)) {
        // Nova format: extract and coerce
        const novaContent = extractNovaContent(responseBody);
        content = coerceNovaToMessageContent(novaContent);
      } else if (isTitanResponse(responseBody)) {
        // Titan format: { results: [{ outputText }] }
        const results = responseBody.results as unknown[];
        if (results?.[0]) {
          const result = results[0] as Record<string, unknown>;
          content = [
            {
              type: "text",
              text: (result.outputText as string) || "",
            },
          ];
        }
      }
    } else if (llm_system === LLMSystem.COHERE) {
      // Cohere: { generations: [{ text }] }
      const generations = responseBody.generations as unknown[];
      if (generations?.[0]) {
        const gen = generations[0] as Record<string, unknown>;
        content = [
          {
            type: "text",
            text: (gen.text as string) || "",
          },
        ];
      }
    } else if (llm_system === LLMSystem.META) {
      // Meta: { generations: [{ text }] }
      const generations = responseBody.generations as unknown[];
      if (generations?.[0]) {
        const gen = generations[0] as Record<string, unknown>;
        content = [
          {
            type: "text",
            text: (gen.text as string) || "",
          },
        ];
      }
    } else if (llm_system === LLMSystem.MISTRALAI) {
      // Mistral: { generations: [{ text }] }
      const generations = responseBody.generations as unknown[];
      if (generations?.[0]) {
        const gen = generations[0] as Record<string, unknown>;
        content = [
          {
            type: "text",
            text: (gen.text as string) || "",
          },
        ];
      }
    }
    return {
      role: role,
      content: content,
    } as BedrockMessage;
  },
  onError: (error) => {
    diag.warn("Error normalizing content blocks:", error);
    return {
      role: "assistant",
      content: [],
    } as BedrockMessage;
  },
});