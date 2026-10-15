import type Anthropic from "@anthropic-ai/sdk";
import type { Attributes } from "@opentelemetry/api";

import type { TraceConfig } from "@arizeai/openinference-core";
import { getOutputAttributes } from "@arizeai/openinference-core";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { getAnthropicOutputMessagesAttributes, isToolUseContentBlock } from "./attributes/content";
import {
  getAnthropicResponseMetadata,
  getAnthropicMetadataAttributes,
  getAnthropicUsageAttributes,
  getAnthropicReasoningTokenAttributes,
  mergeAnthropicUsage,
} from "./attributes/response";
import type {
  Message,
  ContentBlock,
  RawMessageStreamEvent,
  MessageUsageCounts,
  MetadataSources,
} from "./types";
import { toJSONOutput } from "./utils";

/**
 * Mutable state accumulated while consuming an Anthropic message stream.
 *
 * The message is rebuilt from the events as the SDK's own `MessageStream`
 * does, so a stream is recorded with the same attributes and the same JSON
 * output as the message a non-streaming request returns.
 *
 * Usage is reported per attempt: message_start describes the first attempt,
 * which on a server-side fallback stream is the one that declined. Each source
 * is captured on its own so the precedence between them is stated once, where
 * they are merged in {@link getAnthropicStreamAttributes}.
 */
interface AnthropicStreamState {
  /** The message of `message_start`, without its content */
  message?: Omit<Message, "content">;
  /** The content blocks, by their stream `index` */
  content: ContentBlock[];
  /**
   * The raw JSON chunks streamed so far for the input of each tool call not
   * yet stopped, by block index. They are joined and parsed once, when the
   * block stops or the stream ends, as parsing on every delta is quadratic.
   */
  partialInputs: Map<number, string[]>;
  /** The fields `message_delta` events set on the message */
  messageDelta: Record<string, unknown>;
  /** Every usage field reported by `message_delta` events, latest first */
  deltaUsageReport: Record<string, unknown>;
  startUsage: MessageUsageCounts;
  deltaUsage: MessageUsageCounts;
  servingUsage?: MessageUsageCounts;
  responseModel?: string;
  finishReason?: string;
}

/**
 * Applies a `message_delta` event, capturing usage, finish reason and any
 * server-side fallback hop.
 */
function applyAnthropicMessageDelta({
  chunk,
  state,
}: {
  chunk: Extract<RawMessageStreamEvent, { type: "message_delta" }>;
  state: AnthropicStreamState;
}) {
  const usage = chunk.usage ?? {};
  state.deltaUsage = mergeAnthropicUsage(state.deltaUsage, usage);
  state.deltaUsageReport = { ...state.deltaUsageReport, ...withoutNullValues(usage) };
  state.messageDelta = { ...state.messageDelta, ...withoutNullValues(chunk.delta ?? {}) };
  if ("context_management" in chunk && chunk.context_management != null) {
    state.messageDelta.context_management = chunk.context_management;
  }
  if ("input_transformations" in chunk && chunk.input_transformations != null) {
    state.messageDelta.input_transformations = chunk.input_transformations;
  }
  if (chunk.delta?.stop_reason != null) {
    state.finishReason = chunk.delta.stop_reason;
  }
  if (!("iterations" in usage) || usage.iterations == null) {
    return;
  }
  for (const iteration of usage.iterations) {
    if (iteration.type === "fallback_message") {
      state.responseModel = iteration.model;
      state.servingUsage = iteration;
    }
  }
}

/**
 * Drops the fields of a delta that are null, which a stream sends for the
 * fields an event leaves unchanged.
 */
function withoutNullValues(record: object): Record<string, unknown> {
  return Object.fromEntries(Object.entries(record).filter(([, value]) => value != null));
}

/**
 * Applies a `content_block_start` event, adding a copy of the block, so the
 * events the caller receives are never modified.
 */
function applyAnthropicContentBlockStart({
  chunk,
  state,
}: {
  chunk: Extract<RawMessageStreamEvent, { type: "content_block_start" }>;
  state: AnthropicStreamState;
}) {
  const contentBlock = chunk.content_block;
  state.content[chunk.index] = { ...contentBlock };
  if (contentBlock.type === "fallback") {
    state.responseModel = contentBlock.to?.model ?? state.responseModel;
  }
}

/**
 * Applies a `content_block_delta` event to the block it belongs to,
 * accumulating streamed text, reasoning, citations, compaction summaries and
 * tool call arguments.
 */
function applyAnthropicContentBlockDelta({
  chunk,
  state,
}: {
  chunk: Extract<RawMessageStreamEvent, { type: "content_block_delta" }>;
  state: AnthropicStreamState;
}) {
  const block = state.content[chunk.index];
  const delta = chunk.delta;
  if (block == null) {
    return;
  }
  if (delta.type === "text_delta" && block.type === "text") {
    block.text += delta.text;
  } else if (delta.type === "citations_delta" && block.type === "text") {
    Object.assign(block, { citations: [...(block.citations ?? []), delta.citation] });
  } else if (delta.type === "thinking_delta" && block.type === "thinking") {
    block.thinking += delta.thinking;
  } else if (delta.type === "signature_delta" && block.type === "thinking") {
    // As the SDK's message stream, only a thinking block takes a signature.
    block.signature = delta.signature;
  } else if (delta.type === "compaction_delta" && block.type === "compaction") {
    applyAnthropicCompactionDelta(block, delta);
  } else if (delta.type === "input_json_delta" && isToolUseContentBlock(block)) {
    const chunks = state.partialInputs.get(chunk.index);
    if (chunks == null) {
      state.partialInputs.set(chunk.index, [delta.partial_json]);
    } else {
      chunks.push(delta.partial_json);
    }
  }
}

/**
 * Sets the input of each tool call whose JSON is still being accumulated,
 * or just the one of `index`. Input that does not parse, as when the caller
 * left the stream early, is kept as the raw JSON received.
 */
function settleAnthropicToolInputs(state: AnthropicStreamState, index?: number) {
  for (const [blockIndex, chunks] of state.partialInputs) {
    if (index != null && blockIndex !== index) {
      continue;
    }
    const block = state.content[blockIndex];
    if (block != null) {
      // No input JSON is an empty input, as in the SDK's own accumulation.
      const json = chunks.join("");
      Object.assign(block, { input: json === "" ? {} : parseToolInput(json) });
    }
    state.partialInputs.delete(blockIndex);
  }
}

function applyAnthropicCompactionDelta(
  block: Extract<ContentBlock, { type: "compaction" }>,
  delta: Anthropic.Beta.Messages.BetaCompactionContentBlockDelta,
) {
  // As the SDK's message stream: the delta carries the block's final value
  // (null content is a failed compaction), so it replaces rather than appends,
  // and encrypted_content is copied only when the server sent it.
  block.content = delta.content;
  if ("encrypted_content" in delta) {
    block.encrypted_content = delta.encrypted_content;
  }
}

function parseToolInput(partialInput: string): unknown {
  try {
    return JSON.parse(partialInput);
  } catch {
    return partialInput;
  }
}

/**
 * Rebuilds the message a stream delivered, as far as it was read.
 */
function getAnthropicStreamMessage(state: AnthropicStreamState): Record<string, unknown> {
  // On a server-side fallback stream the serving hop's token counts replace
  // message_start's, which belong to the attempt that declined. Only the
  // counts are taken from the iteration, not its `type` and `model`, and the
  // other usage fields (service tier, cache creation...) are kept.
  const servingCounts =
    state.servingUsage != null
      ? withoutNullValues({
          input_tokens: state.servingUsage.input_tokens,
          output_tokens: state.servingUsage.output_tokens,
          cache_creation_input_tokens: state.servingUsage.cache_creation_input_tokens,
          cache_read_input_tokens: state.servingUsage.cache_read_input_tokens,
        })
      : {};
  const startUsage = { ...state.message?.usage, ...servingCounts };
  return {
    ...state.message,
    ...(state.responseModel != null && { model: state.responseModel }),
    content: state.content.filter((block) => block != null),
    ...state.messageDelta,
    usage: { ...startUsage, ...state.deltaUsageReport },
  };
}

/**
 * Builds the span attributes for a consumed Anthropic message stream.
 */
export function getAnthropicStreamAttributes(
  state: AnthropicStreamState,
  metadataSources: MetadataSources,
  traceConfig: TraceConfig,
): Attributes {
  settleAnthropicToolInputs(state);
  const content = state.content.filter((block) => block != null);
  const message = getAnthropicStreamMessage(state);
  const attributes: Attributes = {
    ...getAnthropicOutputMessagesAttributes(
      { role: state.message?.role ?? "assistant", content },
      traceConfig,
    ),
    ...getAnthropicMetadataAttributes({
      metadata: getAnthropicResponseMetadata(
        { ...message, content },
        traceConfig,
        metadataSources.requestToolCalls,
      ),
      contextMetadata: metadataSources.context,
    }),
  };
  if (message.usage != null && typeof message.usage === "object") {
    Object.assign(attributes, getAnthropicReasoningTokenAttributes(message.usage));
  }

  Object.assign(attributes, getOutputAttributes(toJSONOutput(message)));

  if (state.responseModel != null) {
    // Override the model from the value sent by the server
    attributes[SemanticConventions.LLM_MODEL_NAME] = state.responseModel;
    attributes[SemanticConventions.LLM_RESPONSE_MODEL_NAME] = state.responseModel;
  }

  if (state.finishReason != null) {
    attributes[SemanticConventions.LLM_FINISH_REASON] = state.finishReason;
  }

  // On a server-side fallback stream the serving hop's usage replaces
  // message_start's, which belongs to the attempt that declined. The final
  // message_delta then wins field by field, so prompt, completion and cache
  // counts all describe the same model. Prompt and total are derived only
  // after merging so they stay consistent with the cache counts.
  const usage = mergeAnthropicUsage(state.servingUsage ?? state.startUsage, state.deltaUsage);
  Object.assign(attributes, getAnthropicUsageAttributes(usage));

  return attributes;
}

/**
 * Creates the empty state a stream's events are accumulated into.
 */
export function createAnthropicStreamState(): AnthropicStreamState {
  return {
    content: [],
    partialInputs: new Map(),
    messageDelta: {},
    deltaUsageReport: {},
    startUsage: {},
    deltaUsage: {},
  };
}

/**
 * Applies one stream event to the accumulated state.
 */
export function applyAnthropicStreamEvent({
  chunk,
  state,
}: {
  chunk: RawMessageStreamEvent;
  state: AnthropicStreamState;
}) {
  if (chunk.type === "message_start") {
    const { content, ...message } = chunk.message;
    state.message = message;
    // A message can start with content and a stop reason already in it: the
    // stream resuming a programmatic tool call sends its tool_use there, with
    // no block or message_delta events. The SDK's MessageStream keeps them, so
    // they seed the message here too.
    if (Array.isArray(content)) {
      content.forEach((block, index) => {
        if (block != null && typeof block === "object") {
          state.content[index] = { ...block };
        }
      });
    }
    if (chunk.message.stop_reason != null) {
      state.finishReason = chunk.message.stop_reason;
    }
    state.responseModel = chunk.message.model;
    state.startUsage = chunk.message.usage ?? {};
  } else if (chunk.type === "message_delta") {
    applyAnthropicMessageDelta({ chunk, state });
  } else if (chunk.type === "content_block_start") {
    applyAnthropicContentBlockStart({ chunk, state });
  } else if (chunk.type === "content_block_delta") {
    applyAnthropicContentBlockDelta({ chunk, state });
  } else if (chunk.type === "content_block_stop") {
    settleAnthropicToolInputs(state, chunk.index);
  }
}
