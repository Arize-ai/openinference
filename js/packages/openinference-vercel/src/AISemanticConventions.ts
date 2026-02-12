import { ValueOf } from "./typeUtils";

/**
 * Semantic conventions for the Vercel AI SDK `ai.*` attributes.
 *
 * Note: AI SDK v6 also emits `gen_ai.*` (OTel GenAI semantic conventions) which are
 * handled by `@arizeai/openinference-genai`. This file is intentionally scoped to
 * Vercel-specific `ai.*` attributes used as a supplement/fallback.
 *
 * @see https://sdk.vercel.ai/docs/ai-sdk-core/telemetry#collected-data
 */
const AI_PREFIX = "ai" as const;

const AIPrefixes = {
  settings: "settings",
  model: "model",
  usage: "usage",
  telemetry: "telemetry",
  prompt: "prompt",
  toolCall: "toolCall",
  response: "response",
} as const;

const AIUsagePostfixes = {
  completionTokens: "completionTokens",
  promptTokens: "promptTokens",
  inputTokens: "inputTokens",
  outputTokens: "outputTokens",
  totalTokens: "totalTokens",
  reasoningTokens: "reasoningTokens",
  cachedInputTokens: "cachedInputTokens",
  tokens: "tokens", // For embeddings
} as const;

const AIResponsePostfixes = {
  text: "text",
  toolCalls: "toolCalls",
  object: "object",
  finishReason: "finishReason",
  msToFirstChunk: "msToFirstChunk",
  msToFinish: "msToFinish",
  avgOutputTokensPerSecond: "avgOutputTokensPerSecond",
  id: "id",
  model: "model",
  timestamp: "timestamp",
  providerMetadata: "providerMetadata",
} as const;

const AIPromptPostfixes = {
  messages: "messages",
  tools: "tools",
  toolChoice: "toolChoice",
} as const;

const AIToolCallPostfixes = {
  id: "id",
  name: "name",
  args: "args",
  result: "result",
} as const;

// Core settings and identification
const OPERATION_ID = `${AI_PREFIX}.operationId` as const;
const SETTINGS = `${AI_PREFIX}.${AIPrefixes.settings}` as const;
const MODEL_ID = `${AI_PREFIX}.${AIPrefixes.model}.id` as const;
const MODEL_PROVIDER = `${AI_PREFIX}.${AIPrefixes.model}.provider` as const;
const METADATA = `${AI_PREFIX}.${AIPrefixes.telemetry}.metadata` as const;
const FUNCTION_ID = `${AI_PREFIX}.${AIPrefixes.telemetry}.functionId` as const;

// Token counts (fallback if gen_ai.* not present)
const TOKEN_COUNT_COMPLETION =
  `${AI_PREFIX}.${AIPrefixes.usage}.${AIUsagePostfixes.completionTokens}` as const;
const TOKEN_COUNT_INPUT =
  `${AI_PREFIX}.${AIPrefixes.usage}.${AIUsagePostfixes.inputTokens}` as const;
const TOKEN_COUNT_PROMPT =
  `${AI_PREFIX}.${AIPrefixes.usage}.${AIUsagePostfixes.promptTokens}` as const;
const TOKEN_COUNT_OUTPUT =
  `${AI_PREFIX}.${AIPrefixes.usage}.${AIUsagePostfixes.outputTokens}` as const;
const TOKEN_COUNT_TOTAL =
  `${AI_PREFIX}.${AIPrefixes.usage}.${AIUsagePostfixes.totalTokens}` as const;
const TOKEN_COUNT_REASONING =
  `${AI_PREFIX}.${AIPrefixes.usage}.${AIUsagePostfixes.reasoningTokens}` as const;
const TOKEN_COUNT_CACHED_INPUT =
  `${AI_PREFIX}.${AIPrefixes.usage}.${AIUsagePostfixes.cachedInputTokens}` as const;
const TOKEN_COUNT_TOKENS =
  `${AI_PREFIX}.${AIPrefixes.usage}.${AIUsagePostfixes.tokens}` as const;

// Response attributes
const RESPONSE_TEXT =
  `${AI_PREFIX}.${AIPrefixes.response}.${AIResponsePostfixes.text}` as const;
const RESPONSE_TOOL_CALLS =
  `${AI_PREFIX}.${AIPrefixes.response}.${AIResponsePostfixes.toolCalls}` as const;
const RESPONSE_OBJECT =
  `${AI_PREFIX}.${AIPrefixes.response}.${AIResponsePostfixes.object}` as const;
const RESPONSE_FINISH_REASON =
  `${AI_PREFIX}.${AIPrefixes.response}.${AIResponsePostfixes.finishReason}` as const;
const RESPONSE_MS_TO_FIRST_CHUNK =
  `${AI_PREFIX}.${AIPrefixes.response}.${AIResponsePostfixes.msToFirstChunk}` as const;
const RESPONSE_MS_TO_FINISH =
  `${AI_PREFIX}.${AIPrefixes.response}.${AIResponsePostfixes.msToFinish}` as const;
const RESPONSE_AVG_OUTPUT_TOKENS_PER_SECOND =
  `${AI_PREFIX}.${AIPrefixes.response}.${AIResponsePostfixes.avgOutputTokensPerSecond}` as const;
const RESPONSE_ID =
  `${AI_PREFIX}.${AIPrefixes.response}.${AIResponsePostfixes.id}` as const;
const RESPONSE_MODEL =
  `${AI_PREFIX}.${AIPrefixes.response}.${AIResponsePostfixes.model}` as const;
const RESPONSE_TIMESTAMP =
  `${AI_PREFIX}.${AIPrefixes.response}.${AIResponsePostfixes.timestamp}` as const;
const RESPONSE_PROVIDER_METADATA =
  `${AI_PREFIX}.${AIPrefixes.response}.${AIResponsePostfixes.providerMetadata}` as const;

// Prompt/Input attributes
const PROMPT = `${AI_PREFIX}.${AIPrefixes.prompt}` as const;
const PROMPT_MESSAGES = `${PROMPT}.${AIPromptPostfixes.messages}` as const;
const PROMPT_TOOLS = `${PROMPT}.${AIPromptPostfixes.tools}` as const;
const PROMPT_TOOL_CHOICE = `${PROMPT}.${AIPromptPostfixes.toolChoice}` as const;

// Embedding attributes
const EMBEDDING_TEXT = `${AI_PREFIX}.value` as const;
const EMBEDDING_VECTOR = `${AI_PREFIX}.embedding` as const;
const EMBEDDING_TEXTS = `${AI_PREFIX}.values` as const;
const EMBEDDING_VECTORS = `${AI_PREFIX}.embeddings` as const;

// Tool call span attributes
const TOOL_CALL_ID =
  `${AI_PREFIX}.${AIPrefixes.toolCall}.${AIToolCallPostfixes.id}` as const;
const TOOL_CALL_NAME =
  `${AI_PREFIX}.${AIPrefixes.toolCall}.${AIToolCallPostfixes.name}` as const;
const TOOL_CALL_ARGS =
  `${AI_PREFIX}.${AIPrefixes.toolCall}.${AIToolCallPostfixes.args}` as const;
const TOOL_CALL_RESULT =
  `${AI_PREFIX}.${AIPrefixes.toolCall}.${AIToolCallPostfixes.result}` as const;

/**
 * The semantic conventions used by the Vercel AI SDK (`ai.*` attributes).
 * @see https://sdk.vercel.ai/docs/ai-sdk-core/telemetry#collected-data
 */
export const AISemanticConventions = {
  // Core identification
  OPERATION_ID,
  MODEL_ID,
  MODEL_PROVIDER,
  METADATA,
  FUNCTION_ID,
  SETTINGS,

  // Token counts (fallback)
  TOKEN_COUNT_COMPLETION,
  TOKEN_COUNT_PROMPT,
  TOKEN_COUNT_INPUT,
  TOKEN_COUNT_OUTPUT,
  TOKEN_COUNT_TOTAL,
  TOKEN_COUNT_REASONING,
  TOKEN_COUNT_CACHED_INPUT,
  TOKEN_COUNT_TOKENS,

  // Response
  RESPONSE_TEXT,
  RESPONSE_TOOL_CALLS,
  RESPONSE_OBJECT,
  RESPONSE_FINISH_REASON,
  RESPONSE_MS_TO_FIRST_CHUNK,
  RESPONSE_MS_TO_FINISH,
  RESPONSE_AVG_OUTPUT_TOKENS_PER_SECOND,
  RESPONSE_ID,
  RESPONSE_MODEL,
  RESPONSE_TIMESTAMP,
  RESPONSE_PROVIDER_METADATA,

  // Prompt
  PROMPT,
  PROMPT_MESSAGES,
  PROMPT_TOOLS,
  PROMPT_TOOL_CHOICE,

  // Embeddings
  EMBEDDING_TEXT,
  EMBEDDING_VECTOR,
  EMBEDDING_TEXTS,
  EMBEDDING_VECTORS,

  // Tool calls
  TOOL_CALL_ID,
  TOOL_CALL_NAME,
  TOOL_CALL_ARGS,
  TOOL_CALL_RESULT,
} as const;

export const AISemanticConventionsList = Object.freeze(
  Object.values(AISemanticConventions),
);

export type AISemanticConvention = ValueOf<typeof AISemanticConventions>;

// Preferred, explicit naming for v6+ docs/code.
export const VercelAISemanticConventions = AISemanticConventions;
export const VercelAISemanticConventionsList = AISemanticConventionsList;
export type VercelAISemanticConvention = AISemanticConvention;
