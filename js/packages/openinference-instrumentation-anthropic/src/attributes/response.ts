import type { Attributes } from "@opentelemetry/api";
import { diag } from "@opentelemetry/api";

import type { Metadata, TraceConfig } from "@arizeai/openinference-core";
import { getMetadataAttributes, getOutputAttributes } from "@arizeai/openinference-core";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { MODULE_NAME } from "../constants";
import type {
  MessageRequestParams,
  Message,
  ContentBlock,
  ContentBlockParam,
  MessageUsageCounts,
  MetadataSources,
  ToolCallMetadata,
  AnthropicResponseDetails,
} from "../types";
import { toJSONOutput } from "../utils";
import { getAnthropicOutputMessagesAttributes, isToolUseContentBlock } from "./content";

/**
 * Records the reason the model stopped generating tokens. `"refusal"` is the
 * signal that a safety classifier declined the request.
 *
 * @see https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback
 */
function getAnthropicFinishReasonAttributes(stopReason: string | null | undefined): Attributes {
  if (stopReason == null) {
    return {};
  }
  return { [SemanticConventions.LLM_FINISH_REASON]: stopReason };
}

/**
 * Builds the response attributes of a non-streaming Anthropic message.
 */
export function getAnthropicResponseAttributes(
  result: Message,
  metadataSources: MetadataSources,
  traceConfig: TraceConfig,
): Attributes {
  const attributes: Attributes = {
    // Override the model from the value sent by the server
    [SemanticConventions.LLM_MODEL_NAME]: result.model,
    [SemanticConventions.LLM_RESPONSE_MODEL_NAME]: result.model,
    ...getAnthropicFinishReasonAttributes(result.stop_reason),
    ...getAnthropicOutputMessagesAttributes(result, traceConfig),
    ...getAnthropicUsageAttributes(result.usage ?? {}),
    ...getAnthropicReasoningTokenAttributes(result.usage ?? {}),
    ...getAnthropicMetadataAttributes({
      metadata: getAnthropicResponseMetadata(result, traceConfig, metadataSources.requestToolCalls),
      contextMetadata: metadataSources.context,
    }),
  };
  Object.assign(attributes, getOutputAttributes(toJSONOutput(result)));
  return attributes;
}

/**
 * Collects the response details that have no semantic convention of their
 * own, for the span's `metadata`:
 *
 * - `stop_details`: why the model stopped, e.g. the category of a refusal;
 * - `cache_miss_reason`: why the prompt cache was not read;
 * - `container_id`: the container a code execution tool ran in;
 * - `context_management_applied_edits`: the context edits applied to the request;
 * - `input_transformations`: the input blocks the API dropped or let through;
 * - `usage_iterations`: the usage of each sampling iteration. A compaction
 *   iteration's tokens are not part of the top-level usage, so they are not
 *   added to the token counts either;
 * - `server_tool_use`: the number of server tool requests by tool;
 * - `citations`: the citations of each text block, by content index;
 * - `mcp_server_names`: the MCP server of each `mcp_tool_use` call, by id;
 * - `tool_callers` and `toolset_names`: the caller (unless the default direct
 *   one) and toolset of each tool call, by id, from the request's messages and
 *   the response, whose entries win.
 */
export function getAnthropicResponseMetadata(
  message: AnthropicResponseDetails,
  traceConfig: TraceConfig,
  requestToolCalls: ToolCallMetadata = createToolCallMetadata(),
): Metadata {
  const toolCalls = createToolCallMetadata();
  for (const block of message.content) {
    collectToolCallMetadata(block, toolCalls);
  }
  const toolCallers = { ...requestToolCalls.tool_callers, ...toolCalls.tool_callers };
  const toolsetNames = { ...requestToolCalls.toolset_names, ...toolCalls.toolset_names };
  return {
    ...getAnthropicContentMetadata(message, traceConfig),
    ...(Object.keys(toolCallers).length > 0 && { tool_callers: toolCallers }),
    ...(Object.keys(toolsetNames).length > 0 && { toolset_names: toolsetNames }),
  };
}

export function safelyGetToolCallMetadata(body: MessageRequestParams): ToolCallMetadata {
  try {
    return getAnthropicRequestToolCallMetadata(body);
  } catch (error) {
    diag.debug(`Failed to read a ${MODULE_NAME} request's tool calls`, error);
    return createToolCallMetadata();
  }
}

function createToolCallMetadata(): ToolCallMetadata {
  return { tool_callers: {}, toolset_names: {} };
}

/**
 * Records the caller and toolset of a tool call block, and the toolset of a
 * tool result. The default `{"type": "direct"}` caller is left out.
 */
function collectToolCallMetadata(block: ContentBlock | ContentBlockParam, into: ToolCallMetadata) {
  if (block == null || typeof block !== "object") {
    return;
  }
  let id: string | undefined;
  if (isToolUseContentBlock(block)) {
    id = block.id;
    const caller: unknown = "caller" in block ? block.caller : undefined;
    const isDirect =
      caller != null && typeof caller === "object" && "type" in caller && caller.type === "direct";
    if (caller != null && !isDirect) {
      into.tool_callers[id] = caller;
    }
  } else if (block.type === "tool_result") {
    id = block.tool_use_id;
  }
  const toolsetName: unknown = "toolset_name" in block ? block.toolset_name : undefined;
  if (id != null && typeof toolsetName === "string") {
    into.toolset_names[id] = toolsetName;
  }
}

/**
 * Collects the caller and toolset of each tool call and tool result in a
 * request's messages.
 */
function getAnthropicRequestToolCallMetadata(body: MessageRequestParams): ToolCallMetadata {
  const toolCalls = createToolCallMetadata();
  const messages = Array.isArray(body?.messages) ? body.messages : [];
  for (const message of messages) {
    if (!Array.isArray(message?.content)) {
      continue;
    }
    for (const block of message.content) {
      try {
        collectToolCallMetadata(block, toolCalls);
      } catch (error) {
        diag.debug(`Failed to read a ${MODULE_NAME} tool call's metadata`, error);
      }
    }
  }
  return toolCalls;
}

function getAnthropicContentMetadata(
  message: AnthropicResponseDetails,
  traceConfig: TraceConfig,
): Metadata {
  const metadata: Metadata = {
    ...getAnthropicResponseDetailsMetadata(message, traceConfig),
    ...getAnthropicUsageMetadata(message.usage),
  };
  const { content } = message;
  const citations: Record<string, unknown> = {};
  const mcpServerNames: Record<string, string> = {};
  // Citations name and quote the cited sources, so they are left out when the
  // input or output messages are hidden. A citation's cited_text quotes the
  // prompt's documents and is part of the output text, so it alone is left out
  // when just the input text or the output text is hidden.
  const hideCitations =
    traceConfig.hideInputs ||
    traceConfig.hideInputMessages ||
    traceConfig.hideOutputs ||
    traceConfig.hideOutputMessages;
  const hideCitedText = traceConfig.hideInputText || hidesOutputText(traceConfig);
  content.forEach((block, index) => {
    if (
      !hideCitations &&
      block.type === "text" &&
      block.citations != null &&
      block.citations.length > 0
    ) {
      citations[String(index)] = hideCitedText
        ? block.citations.map((citation) => omitKey(citation, "cited_text"))
        : block.citations;
    } else if (block.type === "mcp_tool_use") {
      mcpServerNames[block.id] = block.server_name;
    }
  });
  if (Object.keys(citations).length > 0) {
    metadata.citations = citations;
  }
  if (Object.keys(mcpServerNames).length > 0) {
    metadata.mcp_server_names = mcpServerNames;
  }
  return metadata;
}

/**
 * True when the trace config hides the model's output text.
 */
function hidesOutputText(traceConfig: TraceConfig): boolean {
  return traceConfig.hideOutputs || traceConfig.hideOutputMessages || traceConfig.hideOutputText;
}

/**
 * Copies an object without one of its keys.
 */
function omitKey(value: object, key: string): Record<string, unknown> {
  return Object.fromEntries(Object.entries(value).filter(([entryKey]) => entryKey !== key));
}

function getAnthropicResponseDetailsMetadata(
  message: AnthropicResponseDetails,
  traceConfig: TraceConfig,
): Metadata {
  const metadata: Metadata = {};
  const stopDetails = message.stop_details;
  if (stopDetails != null && typeof stopDetails === "object") {
    // A refusal's explanation is model output, hidden with the outputs.
    metadata.stop_details = hidesOutputText(traceConfig)
      ? omitKey(stopDetails, "explanation")
      : stopDetails;
  }
  const diagnostics = message.diagnostics;
  if (
    diagnostics != null &&
    typeof diagnostics === "object" &&
    "cache_miss_reason" in diagnostics &&
    diagnostics.cache_miss_reason != null
  ) {
    metadata.cache_miss_reason = diagnostics.cache_miss_reason;
  }
  const container = message.container;
  if (
    container != null &&
    typeof container === "object" &&
    "id" in container &&
    typeof container.id === "string"
  ) {
    metadata.container_id = container.id;
  }
  const contextManagement = message.context_management;
  if (
    contextManagement != null &&
    typeof contextManagement === "object" &&
    "applied_edits" in contextManagement &&
    Array.isArray(contextManagement.applied_edits) &&
    contextManagement.applied_edits.length > 0
  ) {
    // The same key as the Python instrumentation
    metadata.context_management_applied_edits = contextManagement.applied_edits;
  }
  if (Array.isArray(message.input_transformations) && message.input_transformations.length > 0) {
    metadata.input_transformations = message.input_transformations;
  }
  return metadata;
}

function getAnthropicUsageMetadata(usage: unknown): Metadata {
  const metadata: Metadata = {};
  if (usage == null || typeof usage !== "object") {
    return metadata;
  }
  if ("iterations" in usage && usage.iterations != null) {
    metadata.usage_iterations = usage.iterations;
  }
  // only when a server tool was used, since the API returns zero counts on most responses
  if (
    "server_tool_use" in usage &&
    usage.server_tool_use != null &&
    typeof usage.server_tool_use === "object" &&
    Object.values(usage.server_tool_use).some((count) => typeof count === "number" && count > 0)
  ) {
    metadata.server_tool_use = usage.server_tool_use;
  }
  return metadata;
}

/**
 * Records response metadata merged into the metadata set on the context, so
 * setting `metadata` again does not drop it. Keys the caller set win.
 */
export function getAnthropicMetadataAttributes({
  metadata,
  contextMetadata,
}: {
  metadata: Metadata;
  contextMetadata?: Metadata;
}): Attributes {
  if (Object.keys(metadata).length === 0) {
    return {};
  }
  return getMetadataAttributes({ ...metadata, ...contextMetadata });
}

/**
 * Get usage attributes from Anthropic response.
 *
 * Anthropic reports `input_tokens` exclusive of cached tokens, so the cache
 * read and write counts are added back into the prompt count. The OpenInference
 * spec treats `prompt_details.*` as sub-counts of `llm.token_count.prompt`, so
 * the prompt count has to include them.
 * https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#tracking-cache-performance
 */
export function getAnthropicUsageAttributes(usage: MessageUsageCounts): Attributes {
  const attributes: Attributes = {};
  const cacheWriteTokens = usage.cache_creation_input_tokens ?? 0;
  const cacheReadTokens = usage.cache_read_input_tokens ?? 0;
  const promptTokens =
    usage.input_tokens != null
      ? usage.input_tokens + cacheWriteTokens + cacheReadTokens
      : undefined;

  if (promptTokens != null) {
    attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT] = promptTokens;
  }
  if (usage.output_tokens != null) {
    attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION] = usage.output_tokens;
  }
  if (promptTokens != null && usage.output_tokens != null) {
    attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL] = promptTokens + usage.output_tokens;
  }
  // Anthropic reports these as 0 when caching did not run, so zeros are left
  // out rather than tagging every span with them.
  if (cacheWriteTokens > 0) {
    attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] = cacheWriteTokens;
  }
  if (cacheReadTokens > 0) {
    attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] = cacheReadTokens;
  }
  return attributes;
}

/**
 * Records the output tokens spent on extended thinking, which Anthropic
 * reports as a breakdown of (and so included in) `output_tokens`. Like the
 * cache counts, a 0, reported when thinking did not run, is left out.
 */
export function getAnthropicReasoningTokenAttributes(usage: object): Attributes {
  const details: unknown =
    "output_tokens_details" in usage ? usage.output_tokens_details : undefined;
  if (
    details != null &&
    typeof details === "object" &&
    "thinking_tokens" in details &&
    typeof details.thinking_tokens === "number" &&
    details.thinking_tokens > 0
  ) {
    return {
      [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING]: details.thinking_tokens,
    };
  }
  return {};
}

/**
 * Merges two usage reports field by field, keeping the later value whenever it
 * is present. Stream usage counts are cumulative, so a later event that omits a
 * field keeps the earlier value, and a later 0 replaces an earlier count.
 */
export function mergeAnthropicUsage(
  earlier: MessageUsageCounts,
  later: MessageUsageCounts,
): MessageUsageCounts {
  return {
    input_tokens: later.input_tokens ?? earlier.input_tokens,
    output_tokens: later.output_tokens ?? earlier.output_tokens,
    cache_creation_input_tokens:
      later.cache_creation_input_tokens ?? earlier.cache_creation_input_tokens,
    cache_read_input_tokens: later.cache_read_input_tokens ?? earlier.cache_read_input_tokens,
  };
}
