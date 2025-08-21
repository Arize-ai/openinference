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
  safelyJSONParse,
  safelyJSONStringify,
  isObjectWithStringKeys,
  assertUnreachable,
} from "@arizeai/openinference-core";
import {
  SemanticConventions,
  MimeType,
} from "@arizeai/openinference-semantic-conventions";
import { setSpanAttribute } from "./attribute-helpers";
import {
  ConverseStreamEventData,
  ConverseStreamProcessingState,
  isValidConverseStreamEventData,
  toNormalizedConverseStreamEvent,
} from "../types/bedrock-types";

/**
 * Resolves the target tool use id from either an explicit id or a content block index.
 *
 * @param params.id Optional explicit tool id carried on the event
 * @param params.contentBlockIndex Optional content block index to map into an id
 * @param params.indexMap Optional index->id mapping established at tool start
 * @returns The resolved tool id if available
 */
function resolveToolUseId({
  id,
  contentBlockIndex,
  indexMap,
}: {
  id?: string;
  contentBlockIndex?: number;
  indexMap?: Record<number, string>;
}): string | undefined {
  if (id) return id;
  if (contentBlockIndex !== undefined && indexMap) {
    return indexMap[contentBlockIndex];
  }
}

/**
 * Starts tracking a tool call encountered in a Converse stream.
 *
 * Records the mapping between a content block index and a toolUseId so that
 * subsequent input chunks can be correlated, and initializes a new tool call
 * entry with an empty input buffer for incremental JSON accumulation.
 *
 * @param state Stream processing state to update
 * @param params.id The tool use identifier from the stream event
 * @param params.name The tool name provided by the model
 * @param params.contentBlockIndex Optional index to correlate future input chunks
 */
function startToolCall(
  state: ConverseStreamProcessingState,
  {
    id,
    name,
    contentBlockIndex,
  }: { id: string; name: string; contentBlockIndex?: number },
) {
  if (contentBlockIndex !== undefined) {
    state.toolUseIdByIndex ??= {};
    state.toolUseIdByIndex[contentBlockIndex] = id;
  }
  state.toolCalls.push({ id, name, input: {}, partialJsonInput: "" });
}

/**
 * Appends a partial JSON input chunk for the current tool call.
 *
 * Resolves the target tool call either by explicit id or by the content block
 * index mapping established at tool start. Accumulates the chunk into a
 * string buffer and attempts to parse it into a JSON object; on parse errors,
 * accumulation continues until valid JSON is formed by subsequent chunks.
 *
 * @param state Stream processing state to update
 * @param params.chunk Partial JSON string for tool input
 * @param params.contentBlockIndex Optional index to resolve the tool id
 * @param params.id Optional explicit tool id
 */
function appendToolInputChunk(
  state: ConverseStreamProcessingState,
  {
    chunk,
    contentBlockIndex,
    id,
  }: { chunk: string; contentBlockIndex?: number; id?: string },
) {
  const targetId = resolveToolUseId({
    id,
    contentBlockIndex,
    indexMap: state.toolUseIdByIndex,
  });
  if (!targetId) return;
  const tool = state.toolCalls.find((t) => t.id === targetId);
  if (!tool) return;

  tool.partialJsonInput = (tool.partialJsonInput ?? "") + chunk;

  const parsed = safelyJSONParse(tool.partialJsonInput);
  if (parsed != null && isObjectWithStringKeys(parsed)) {
    tool.input = parsed;
  }
}

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
): void {
  const ev = toNormalizedConverseStreamEvent(data);
  if (!ev) return;

  switch (ev.kind) {
    case "messageStart":
      return;
    case "messageStop":
      state.stopReason = ev.stopReason;
      return;
    case "textDelta":
      state.outputText += ev.text;
      return;
    case "toolUseStart":
      startToolCall(state, ev);
      return;
    case "toolUseInputChunk":
      appendToolInputChunk(state, ev);
      return;
    case "metadata":
      state.usage = {
        inputTokens: ev.usage.inputTokens ?? state.usage.inputTokens,
        outputTokens: ev.usage.outputTokens ?? state.usage.outputTokens,
        totalTokens: ev.usage.totalTokens ?? state.usage.totalTokens,
      };
      return;
    default:
      assertUnreachable(ev);
  }
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

  // Clean up tool calls - drop partialJsonInput from final output
  const cleanedToolCalls = toolCalls.map(({ id, name, input }) => ({
    id,
    name,
    input,
  }));

  const outputValue = {
    text: outputText || "",
    tool_calls: cleanedToolCalls,
    usage: normalizedUsage,
    streaming: true,
    ...(stopReason && { stop_reason: stopReason }),
  };

  // Set output value as JSON (matching converse response behavior)
  setSpanAttribute(
    span,
    SemanticConventions.OUTPUT_VALUE,
    safelyJSONStringify(outputValue),
  );
  setSpanAttribute(span, SemanticConventions.OUTPUT_MIME_TYPE, MimeType.JSON);

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
      safelyJSONStringify(toolCall.input),
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
    const state: ConverseStreamProcessingState = {
      outputText: "",
      toolCalls: [],
      usage: {},
    };

    for await (const chunk of stream) {
      // ConverseStream returns events directly as objects, no need to decode bytes
      if (isValidConverseStreamEventData(chunk)) {
        processConverseStreamChunk(chunk, state);
      }
    }

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
