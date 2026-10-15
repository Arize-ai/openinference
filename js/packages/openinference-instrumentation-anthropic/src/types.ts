import type Anthropic from "@anthropic-ai/sdk";

import type { Metadata } from "@arizeai/openinference-core";

type MessageCreateParams =
  | Parameters<typeof Anthropic.Messages.prototype.create>[0]
  | Parameters<typeof Anthropic.Beta.Messages.prototype.create>[0];

/** The params of a request with messages: a messages create or a token count */
export type MessageRequestParams =
  | MessageCreateParams
  | Parameters<typeof Anthropic.Messages.prototype.countTokens>[0]
  | Parameters<typeof Anthropic.Beta.Messages.prototype.countTokens>[0];

export type TokensCount =
  | Anthropic.Messages.MessageTokensCount
  | Anthropic.Beta.Messages.BetaMessageTokensCount;

export type ToolRunnerParams = Parameters<typeof Anthropic.Beta.Messages.prototype.toolRunner>[0];

export type ToolRunner = ReturnType<typeof Anthropic.Beta.Messages.prototype.toolRunner>;

export type RunnableTool = Extract<ToolRunnerParams["tools"][number], { parse: unknown }>;

export type MessageParam =
  | Anthropic.Messages.MessageParam
  | Anthropic.Beta.Messages.BetaMessageParam;

export type Message = Anthropic.Messages.Message | Anthropic.Beta.Messages.BetaMessage;

export type ContentBlock =
  | Anthropic.Messages.ContentBlock
  | Anthropic.Beta.Messages.BetaContentBlock;

export type ContentBlockParam =
  | Anthropic.Messages.ContentBlockParam
  | Anthropic.Beta.Messages.BetaContentBlockParam;

export type ToolUseContentBlock = Extract<
  ContentBlock | ContentBlockParam,
  { type: "tool_use" | "server_tool_use" | "mcp_tool_use" }
>;

export type ImageBlockParam =
  | Anthropic.Messages.ImageBlockParam
  | Anthropic.Beta.Messages.BetaImageBlockParam;

export type ToolDefinition = NonNullable<MessageRequestParams["tools"]>[number];

/**
 * The parts of a message the output attributes are read from, shared by a
 * parsed message and one rebuilt from a stream.
 */
export type OutputMessage = Pick<Message, "role"> & { content: ContentBlock[] };

export type RawMessageStreamEvent =
  | Anthropic.Messages.RawMessageStreamEvent
  | Anthropic.Beta.Messages.BetaRawMessageStreamEvent;

type MessageUsage =
  | Anthropic.Messages.Usage
  | Anthropic.Messages.MessageDeltaUsage
  | Anthropic.Beta.Messages.BetaUsage
  | Anthropic.Beta.Messages.BetaMessageDeltaUsage
  | Anthropic.Beta.Messages.BetaFallbackMessageIterationUsage;

export type MessageUsageCounts = Partial<
  Pick<
    MessageUsage,
    "input_tokens" | "output_tokens" | "cache_creation_input_tokens" | "cache_read_input_tokens"
  >
>;

/**
 * What a response's metadata is merged with: the metadata set on the context
 * (whose keys win) and the tool call details read from the request.
 */
export interface MetadataSources {
  context?: Metadata;
  requestToolCalls: ToolCallMetadata;
}

/**
 * The caller and toolset of each tool call, by tool call id.
 */
export interface ToolCallMetadata {
  tool_callers: Record<string, unknown>;
  toolset_names: Record<string, string>;
}

/**
 * The response fields metadata is read from, shared by a parsed message and
 * one rebuilt from a stream, whose fields are only known once they arrive.
 */
export type AnthropicResponseDetails = { content: ContentBlock[]; usage?: unknown } & Partial<
  Record<
    "stop_details" | "diagnostics" | "container" | "context_management" | "input_transformations",
    unknown
  >
>;
