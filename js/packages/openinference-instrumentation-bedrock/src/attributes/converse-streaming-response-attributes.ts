/**
 * Streaming response attribute extraction for AWS Bedrock Converse API instrumentation
 *
 * Handles extraction of semantic convention attributes from streaming Converse responses including:
 * - Streaming content processing for Converse events
 * - Tool call extraction from streams
 * - Token usage accumulation from metadata events
 * - Real-time span attribute setting
 * - Safe stream splitting with original stream preservation
 */

import { Span, diag } from "@opentelemetry/api";
import {
  withSafety,
  isObjectWithStringKeys,
} from "@arizeai/openinference-core";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";
import { setSpanAttribute } from "./attribute-helpers";
import {
  ConverseStreamEventData,
  ConverseStreamProcessingState,
  isValidConverseStreamEventData,
  isConverseMessageStartEvent,
  isConverseMessageStopEvent,
  isConverseContentBlockStartEvent,
  isConverseContentBlockDeltaEvent,
  isConverseContentBlockStopEvent,
  isConverseMetadataEvent,
} from "../types/bedrock-types";
import { safelySplitStream } from "./invoke-model-streaming-response-attributes";

/**
 * Processes Converse stream chunks and updates processing state
 * Handles messageStart, contentBlockDelta, toolUse events, and metadata with proper accumulation
 *
 * @param data The parsed Converse stream event data
 * @param state Current processing state to update
 * @returns Updated processing state
 */
function processConverseStreamChunk(
  data: ConverseStreamEventData,
  state: ConverseStreamProcessingState,
): ConverseStreamProcessingState {
  // Handle message start events
  if (isConverseMessageStartEvent(data)) {
    // Message start just indicates the beginning of a message, no data to accumulate
    // The role is already known from the response context (always "assistant")
  }

  // Handle content block start events (primarily for tool use)
  if (isConverseContentBlockStartEvent(data) && data.contentBlockStart) {
    const toolUse = data.contentBlockStart.start?.toolUse;
    if (toolUse && toolUse.toolUseId && toolUse.name) {
      // Start tracking a new tool call
      state.toolCalls.push({
        id: toolUse.toolUseId,
        name: toolUse.name,
        input: {}, // Will be filled in by delta events
      });
    }
  }

  // Handle content block delta events (text content and tool input)
  if (isConverseContentBlockDeltaEvent(data) && data.contentBlockDelta) {
    const delta = data.contentBlockDelta.delta;

    // Accumulate text content
    if (delta && delta.text) {
      state.outputText += delta.text;
    }

    // Accumulate tool use input (JSON string chunks)
    if (delta && delta.toolUse?.input) {
      // Find the most recent tool call to update
      const mostRecentTool = state.toolCalls[state.toolCalls.length - 1];
      if (mostRecentTool) {
        // Tool input comes as JSON string chunks, parse when complete
        try {
          // For now, we'll collect the input string and try to parse it
          // In a real implementation, we might need to buffer partial JSON
          const inputStr = delta.toolUse.input;
          const parsedInput = JSON.parse(inputStr);
          mostRecentTool.input = { ...mostRecentTool.input, ...parsedInput };
        } catch (error) {
          // If JSON is partial/invalid, we'll try again with next chunk
          // For now, store as raw string
          if (
            typeof mostRecentTool.input === "object" &&
            mostRecentTool.input !== null
          ) {
            (mostRecentTool.input as any)._raw =
              (mostRecentTool.input as any)._raw || "";
            (mostRecentTool.input as any)._raw += delta.toolUse.input;
          }
        }
      }
    }
  }

  // Handle content block stop events
  if (isConverseContentBlockStopEvent(data)) {
    // Content block is complete, no additional processing needed for now
    // Could be used for finalizing tool input parsing if needed
  }

  // Handle message stop events
  if (isConverseMessageStopEvent(data) && data.messageStop) {
    state.stopReason = data.messageStop.stopReason;
  }

  // Handle metadata events (usage information)
  if (isConverseMetadataEvent(data) && data.metadata) {
    const usage = data.metadata.usage;
    if (usage) {
      state.usage = {
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        totalTokens: usage.totalTokens,
      };
    }
  }

  return state;
}

/**
 * Sets streaming output attributes on the OpenTelemetry span for Converse streams
 * Processes accumulated content and usage data into semantic convention attributes
 *
 * @param params Object containing span and accumulated streaming data
 */
function setConverseStreamingOutputAttributes({
  span,
  outputText,
  toolCalls,
  usage,
  stopReason,
}: {
  span: Span;
  outputText: string;
  toolCalls: ConverseStreamProcessingState["toolCalls"];
  usage: ConverseStreamProcessingState["usage"];
  stopReason?: string;
}): void {
  // Create the output value structure similar to converse response format
  // Convert usage from camelCase to snake_case for consistency
  const normalizedUsage = {
    ...(usage.inputTokens !== undefined && { input_tokens: usage.inputTokens }),
    ...(usage.outputTokens !== undefined && {
      output_tokens: usage.outputTokens,
    }),
    ...(usage.totalTokens !== undefined && { total_tokens: usage.totalTokens }),
  };

  const outputValue = {
    text: outputText || "",
    tool_calls: toolCalls,
    usage: normalizedUsage,
    streaming: true,
    ...(stopReason && { stop_reason: stopReason }),
  };

  // Set output value as JSON (matching converse response behavior)
  setSpanAttribute(
    span,
    SemanticConventions.OUTPUT_VALUE,
    JSON.stringify(outputValue),
  );
  setSpanAttribute(
    span,
    SemanticConventions.OUTPUT_MIME_TYPE,
    "application/json",
  );

  // Set the message role (always assistant for converse responses)
  setSpanAttribute(
    span,
    `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`,
    "assistant",
  );

  // Set the main accumulated text content
  if (outputText) {
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`,
      outputText,
    );
  }

  // Set tool call attributes with sequential indexing
  toolCalls.forEach((toolCall, toolCallIndex) => {
    const toolCallPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}`;

    setSpanAttribute(
      span,
      `${toolCallPrefix}.${SemanticConventions.TOOL_CALL_ID}`,
      toolCall.id,
    );
    setSpanAttribute(
      span,
      `${toolCallPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`,
      toolCall.name,
    );
    setSpanAttribute(
      span,
      `${toolCallPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`,
      JSON.stringify(toolCall.input),
    );
  });

  // Set stop reason attribute if available
  if (stopReason) {
    setSpanAttribute(span, "llm.stop_reason", stopReason);
  }

  // Set usage attributes
  if (usage.inputTokens !== undefined) {
    setSpanAttribute(
      span,
      SemanticConventions.LLM_TOKEN_COUNT_PROMPT,
      usage.inputTokens,
    );
  }
  if (usage.outputTokens !== undefined) {
    setSpanAttribute(
      span,
      SemanticConventions.LLM_TOKEN_COUNT_COMPLETION,
      usage.outputTokens,
    );
  }
  if (usage.totalTokens !== undefined) {
    setSpanAttribute(
      span,
      SemanticConventions.LLM_TOKEN_COUNT_TOTAL,
      usage.totalTokens,
    );
  }
}

/**
 * Consumes AWS Bedrock Converse streaming response chunks and extracts attributes for OpenTelemetry span.
 * Supports the unified Converse API streaming format with messageStart, contentBlockDelta, metadata events.
 *
 * This function processes the Converse streaming format and accumulates response content,
 * setting appropriate semantic convention attributes on the provided span. It runs in the
 * background without blocking the user's stream consumption.
 *
 * @param stream - The Converse response stream (AsyncIterable), typically from safelySplitStream()
 * @param span - The OpenTelemetry span to set attributes on
 * @throws {Error} If critical stream processing errors occur
 *
 * @example
 * ```typescript
 * // Background processing after stream splitting
 * const { instrumentationStream, userStream } = safelySplitStream(response.stream);
 * consumeConverseStreamChunks({ stream: instrumentationStream, span })
 *   .then(() => span.end())
 *   .catch(error => { span.recordException(error); span.end(); });
 * ```
 */
export const consumeConverseStreamChunks = withSafety({
  fn: async ({
    stream,
    span,
  }: {
    stream: AsyncIterable<unknown>;
    span: Span;
  }): Promise<void> => {
    // Initialize processing state for Converse streaming
    const state: ConverseStreamProcessingState = {
      outputText: "",
      toolCalls: [],
      usage: {},
    };

    for await (const chunk of stream) {
      // ConverseStream returns events directly as objects, no need to decode bytes
      if (isValidConverseStreamEventData(chunk)) {
        const data: ConverseStreamEventData = chunk;
        processConverseStreamChunk(data, state);
      }
    }

    // Set final streaming output attributes
    setConverseStreamingOutputAttributes({
      span,
      outputText: state.outputText,
      toolCalls: state.toolCalls,
      usage: state.usage,
      stopReason: state.stopReason,
    });
  },
  onError: (error) => {
    diag.warn("Error consuming Converse stream chunks:", error);
    throw error;
  },
});
