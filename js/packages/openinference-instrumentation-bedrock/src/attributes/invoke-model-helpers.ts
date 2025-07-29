import { withSafety } from "@arizeai/openinference-core";
import { InvokeModelCommand, InvokeModelResponse } from "@aws-sdk/client-bedrock-runtime";
import { diag } from "@opentelemetry/api";
import {
  BedrockMessage,
  InvokeModelRequestBody,
  isTextContent,
  MessageContent,
  TextContent,
  ToolUseContent,
} from "../types/bedrock-types";
import { LLMSystem } from "@arizeai/openinference-semantic-conventions";

/**
 * Safely parses the InvokeModel response body with comprehensive error handling
 * Handles multiple response body formats and provides null fallback on error
 *
 * @param response The InvokeModelResponse containing the response body to parse
 * @returns {InvokeModelResponseBody | null} Parsed response body or null on error
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
 * @returns {InvokeModelRequestBody} Parsed request body or fallback structure on error
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
 * @returns {ExtractedInvocationParameters} Object containing extracted parameters
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

/**
 * Coerces Nova-style content blocks to standard MessageContent format
 * Works with raw JSON structure without importing Nova types
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
 */
function extractNovaContent(responseBody: Record<string, unknown>): unknown {
  const output = responseBody.output as Record<string, unknown> | undefined;
  if (!output) return [];

  const message = output.message as Record<string, unknown> | undefined;
  if (!message) return [];

  return message.content || [];
}

function isNovaResponse(responseBody: Record<string, unknown>): boolean {
  return !!(
    responseBody.output &&
    typeof responseBody.output === "object" &&
    (responseBody.output as Record<string, unknown>).message
  );
}

function isTitanResponse(responseBody: Record<string, unknown>): boolean {
  return !!(
    responseBody.results &&
    Array.isArray(responseBody.results) &&
    typeof responseBody.inputTextTokenCount === "number"
  );
}

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
