/**
 * Streaming response attribute extraction for AWS Bedrock InvokeModel instrumentation
 *
 * Handles extraction of semantic convention attributes from streaming InvokeModel responses including:
 * - Streaming content processing
 * - Tool call extraction from streams
 * - Token usage accumulation
 * - Real-time span attribute setting
 * - Safe stream splitting with original stream preservation
 */

import { Span, diag } from "@opentelemetry/api";
import {
  SemanticConventions,
  MimeType,
} from "@arizeai/openinference-semantic-conventions";
import { 
  ContentBlockDelta,
} from "@aws-sdk/client-bedrock-runtime";
import { withSafety, isObjectWithStringKeys } from "@arizeai/openinference-core";
import { ReadableStream } from "node:stream/web";
import { isToolUseContent } from "../types/bedrock-types";
import { setSpanAttribute } from "./attribute-helpers";

/**
 * Interface for raw stream chunks from AWS SDK (network level)
 * 
 * Note: This represents the low-level network transport format that the AWS SDK
 * uses internally. The SDK doesn't expose a specific type for this raw chunk format,
 * which is why we define it here. The actual streaming event types come after
 * parsing these raw bytes.
 */
interface StreamChunk {
  chunk: {
    bytes: Uint8Array;
  };
}

/**
 * Streaming event data that matches Bedrock's raw JSON event format
 * 
 * Note: We define this custom interface because we're parsing the raw JSON lines
 * from the stream bytes directly, before the AWS SDK processes them into its
 * own event types (like ContentBlockDeltaEvent). The SDK types represent the
 * final processed events, but we need to work with the intermediate JSON structure
 * to avoid interfering with the user's stream consumption.
 * 
 * The AWS SDK types are designed for post-processed events that go through the
 * SDK's event stream parser, but we're instrumenting at the raw byte level
 * to preserve the original user stream.
 */
interface StreamEventData {
  type:
    | "message_start"
    | "content_block_start"
    | "content_block_delta"
    | "message_delta";
  message?: {
    usage?: Record<string, number>;
  };
  content_block?: {
    type: string;
    text?: string;
    id?: string;
    name?: string;
    input?: Record<string, unknown>;
  };
  delta?: ContentBlockDelta;
  usage?: Record<string, number>;
}

/**
 * Type guard to validate raw stream chunk structure from AWS SDK
 * Ensures chunk has the expected structure with bytes property
 *
 * @param chunk The chunk object to validate
 * @returns {boolean} True if chunk is a valid StreamChunk with bytes, false otherwise
 */
function isValidStreamChunk(chunk: unknown): chunk is StreamChunk {
  return (
    isObjectWithStringKeys(chunk) &&
    "chunk" in chunk &&
    isObjectWithStringKeys(chunk.chunk) &&
    "bytes" in chunk.chunk &&
    chunk.chunk.bytes instanceof Uint8Array
  );
}

/**
 * Type guard to validate streaming event data structure from Bedrock streams
 * Ensures event data has valid type property matching expected event types
 *
 * @param data The event data object to validate
 * @returns {boolean} True if data is valid StreamEventData, false otherwise
 */
function isValidStreamEventData(data: unknown): data is StreamEventData {
  return (
    isObjectWithStringKeys(data) &&
    "type" in data &&
    typeof data.type === "string" &&
    [
      "message_start",
      "content_block_start",
      "content_block_delta",
      "message_delta",
    ].includes(data.type)
  );
}

/**
 * Processes accumulated streaming content and sets OpenInference span attributes
 * Creates structured output representation and sets semantic convention attributes
 *
 * @param params Object containing accumulated streaming data
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.outputText Accumulated text content from streaming response
 * @param params.contentBlocks Array of content blocks including tool calls
 * @param params.usage Token usage statistics from the streaming response
 */
function setStreamingOutputAttributes({
  span,
  outputText,
  contentBlocks,
  usage,
}: {
  span: Span;
  outputText: string;
  contentBlocks: Array<{
    type: string;
    text?: string;
    id?: string;
    name?: string;
    input?: Record<string, unknown>;
  }>;
  usage: Record<string, number>;
}): void {
  // Create simple output representation with only actual data we have
  const outputValue = {
    text: outputText || "",
    tool_calls: contentBlocks.filter(isToolUseContent).map((content) => ({
      id: content.id,
      name: content.name,
      input: content.input,
    })),
    usage: usage || {},
    streaming: true,
  };

  // Set output value as JSON
  setSpanAttribute(
    span,
    SemanticConventions.OUTPUT_VALUE,
    JSON.stringify(outputValue),
  );
  setSpanAttribute(span, SemanticConventions.OUTPUT_MIME_TYPE, MimeType.JSON);

  // Set structured output message attributes for text content
  if (outputText) {
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`,
      "assistant",
    );
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`,
      outputText,
    );
  }

  // Extract tool call attributes from content blocks
  const toolUseBlocks = contentBlocks.filter(isToolUseContent);
  toolUseBlocks.forEach((content, toolCallIndex) => {
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`,
      content.name,
    );
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`,
      content.input ? JSON.stringify(content.input) : undefined,
    );
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_ID}`,
      content.id,
    );
  });

  // Set usage attributes directly
  if (usage) {
    setSpanAttribute(
      span,
      SemanticConventions.LLM_TOKEN_COUNT_PROMPT,
      usage.input_tokens,
    );
    setSpanAttribute(
      span,
      SemanticConventions.LLM_TOKEN_COUNT_COMPLETION,
      usage.output_tokens,
    );
  }
}

/**
 * Safely splits a stream for instrumentation while preserving the original stream.
 * Uses native Web Streams API for optimal performance and simplicity.
 *
 * @param originalStream - The original response stream to split
 * @returns Object with instrumentationStream (for background processing) and userStream (original data for user)
 *          If splitting fails, returns { userStream: originalStream } to ensure user always gets their stream back
 */
export function safelySplitStream({
  originalStream,
}: {
  originalStream: AsyncIterable<unknown>;
}): {
  instrumentationStream?: AsyncIterable<unknown>;
  userStream: AsyncIterable<unknown>;
} {
  try {
    // Convert async iterable to ReadableStream and use native tee() - just like OpenAI!
    const readableStream = ReadableStream.from(originalStream);
    const [instrumentationStream, userStream] = readableStream.tee();

    return {
      instrumentationStream,
      userStream,
    };
  } catch (error) {
    // Fallback: preserve original stream for user
    return {
      userStream: originalStream,
    };
  }
}

/**
 * Consumes AWS Bedrock streaming response chunks and extracts attributes for OpenTelemetry span.
 *
 * This function processes the Bedrock streaming format which consists of JSON lines
 * containing different event types (message_start, content_block_delta, etc.).
 * It accumulates the response content and sets appropriate semantic convention attributes
 * on the provided span. This function is designed to run in the background without
 * blocking the user's stream consumption.
 *
 * @param stream - The Bedrock response stream (AsyncIterable), typically from splitStream()
 * @param span - The OpenTelemetry span to set attributes on
 * @throws {Error} If critical stream processing errors occur
 *
 * @example
 * ```typescript
 * // Background processing after stream splitting
 * const [instrumentationStream, userStream] = await splitStream(response.body);
 * consumeBedrockStreamChunks(instrumentationStream, span)
 *   .then(() => span.end())
 *   .catch(error => { span.recordException(error); span.end(); });
 * ```
 */
export const consumeBedrockStreamChunks = withSafety({
  fn: async ({
    stream,
    span,
  }: {
    stream: AsyncIterable<unknown>;
    span: Span;
  }): Promise<void> => {
    let outputText = "";
    const contentBlocks: Array<{
      type: string;
      text?: string;
      id?: string;
      name?: string;
      input?: Record<string, unknown>;
    }> = [];
    let usage: Record<string, number> = {};

    for await (const chunk of stream) {
      // Type guard for chunk structure
      if (isValidStreamChunk(chunk)) {
        const text = new TextDecoder().decode(chunk.chunk.bytes);
        const lines = text.split("\n").filter((line) => line.trim());

        for (const line of lines) {
          if (line.trim()) {
            try {
              const rawData = JSON.parse(line);
              if (!isValidStreamEventData(rawData)) {
                continue; // Skip invalid event data
              }
              const data: StreamEventData = rawData;

              // Handle different event types
              if (data.type === "message_start" && data.message) {
                usage = data.message.usage || {};
              }

              if (data.type === "content_block_start" && data.content_block) {
                contentBlocks.push(data.content_block);
              }

              if (data.type === "content_block_delta" && data.delta?.text) {
                // Accumulate text content
                outputText += data.delta.text;

                // Also update the content block for tool processing
                const lastTextBlock = contentBlocks.find(
                  (block) => block.type === "text",
                );
                if (lastTextBlock) {
                  lastTextBlock.text =
                    (lastTextBlock.text || "") + data.delta.text;
                } else {
                  contentBlocks.push({
                    type: "text",
                    text: data.delta.text,
                  });
                }
              }

              if (data.type === "message_delta" && data.usage) {
                usage = { ...usage, ...data.usage };
              }
            } catch (parseError) {
              // Skip malformed JSON lines silently
              continue;
            }
          }
        }
      }
    }

    setStreamingOutputAttributes({ span, outputText, contentBlocks, usage });
  },
  onError: (error) => {
    diag.warn("Error consuming Bedrock stream chunks:", error);
    throw error;
  },
});
