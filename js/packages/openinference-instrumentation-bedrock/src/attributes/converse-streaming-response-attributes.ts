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

import type { Span } from "@opentelemetry/api";
import { diag } from "@opentelemetry/api";

import {
  assertUnreachable,
  isObjectWithStringKeys,
  safelyJSONParse,
  safelyJSONStringify,
  withSafety,
} from "@arizeai/openinference-core";
import { MimeType, SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import type {
  ConverseStreamContentBlock,
  ConverseStreamEventData,
  ConverseStreamProcessingState,
} from "../types/bedrock-types";
import {
  isValidConverseStreamEventData,
  toNormalizedConverseStreamEvent,
} from "../types/bedrock-types";
import { setSpanAttribute, toBase64Bytes } from "./attribute-helpers";

/**
 * Accumulates a single content-block delta into the ordered contentBlocksByIndex map.
 * Falls back to appending at the end when no index is provided by the stream event.
 */
function mergeContentBlock(
  state: ConverseStreamProcessingState,
  contentBlockIndex: number | undefined,
  merge: (existing: ConverseStreamContentBlock | undefined) => ConverseStreamContentBlock,
): void {
  state.contentBlocksByIndex ??= {};
  const index = contentBlockIndex ?? Object.keys(state.contentBlocksByIndex).length;
  state.contentBlocksByIndex[index] = merge(state.contentBlocksByIndex[index]);
}

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
  { id, name, contentBlockIndex }: { id: string; name: string; contentBlockIndex?: number },
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
  { chunk, contentBlockIndex, id }: { chunk: string; contentBlockIndex?: number; id?: string },
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
      mergeContentBlock(state, ev.contentBlockIndex, (existing) => ({
        type: "text",
        text: (existing?.type === "text" ? existing.text : "") + ev.text,
      }));
      return;
    case "reasoningDelta":
      mergeContentBlock(state, ev.contentBlockIndex, (existing) => {
        const base = existing?.type === "reasoning" ? existing : { type: "reasoning" as const };
        const redactedContentBytes =
          ev.redactedContent !== undefined
            ? Buffer.concat([base.redactedContentBytes ?? Buffer.alloc(0), ev.redactedContent])
            : base.redactedContentBytes;
        return {
          ...base,
          ...(ev.text !== undefined && { text: (base.text ?? "") + ev.text }),
          ...(ev.signature !== undefined && { signature: ev.signature }),
          ...(redactedContentBytes !== undefined && {
            redactedContentBytes,
            data: toBase64Bytes(redactedContentBytes) ?? base.data,
          }),
        };
      });
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
  contentBlocksByIndex,
}: {
  span: Span;
  outputText: string;
  toolCalls: ConverseStreamProcessingState["toolCalls"];
  usage: ConverseStreamProcessingState["usage"];
  stopReason?: string;
  contentBlocksByIndex: ConverseStreamProcessingState["contentBlocksByIndex"];
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
  setSpanAttribute(span, SemanticConventions.OUTPUT_VALUE, safelyJSONStringify(outputValue));
  setSpanAttribute(span, SemanticConventions.OUTPUT_MIME_TYPE, MimeType.JSON);

  // Set the message role (always assistant for converse responses)
  setSpanAttribute(
    span,
    `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`,
    "assistant",
  );

  // Set ordered content blocks (text/reasoning), preserving stream block order
  const orderedContentBlocks = Object.entries(contentBlocksByIndex ?? {})
    .sort(([a], [b]) => Number(a) - Number(b))
    .map(([, block]) => block);

  orderedContentBlocks.forEach((block, contentBlockIndex) => {
    const contentPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENTS}.${contentBlockIndex}`;
    setSpanAttribute(
      span,
      `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`,
      block.type,
    );
    if (block.text) {
      setSpanAttribute(
        span,
        `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`,
        block.text,
      );
    }
    if (block.type === "reasoning" && block.signature) {
      setSpanAttribute(
        span,
        `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_SIGNATURE}`,
        block.signature,
      );
    }
    if (block.type === "reasoning" && block.data) {
      setSpanAttribute(
        span,
        `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_DATA}`,
        block.data,
      );
    }
  });

  // Set tool call attributes with sequential indexing
  toolCalls.forEach((toolCall, toolCallIndex) => {
    const toolCallPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}`;

    setSpanAttribute(span, `${toolCallPrefix}.${SemanticConventions.TOOL_CALL_ID}`, toolCall.id);
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
    setSpanAttribute(span, SemanticConventions.LLM_TOKEN_COUNT_PROMPT, usage.inputTokens);
  }
  if (usage.outputTokens !== undefined) {
    setSpanAttribute(span, SemanticConventions.LLM_TOKEN_COUNT_COMPLETION, usage.outputTokens);
  }
  if (usage.totalTokens !== undefined) {
    setSpanAttribute(span, SemanticConventions.LLM_TOKEN_COUNT_TOTAL, usage.totalTokens);
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
  fn: async ({ stream, span }: { stream: AsyncIterable<unknown>; span: Span }): Promise<void> => {
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
      contentBlocksByIndex: state.contentBlocksByIndex,
    });
  },
  onError: (error) => {
    diag.warn("Error consuming Converse stream chunks:", error);
    throw error;
  },
});
