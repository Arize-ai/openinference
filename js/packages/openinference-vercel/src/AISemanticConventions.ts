import { ValueOf } from "./typeUtils";

/**
 * Below are the semantic conventions for @vercel/otel instrumentation.
 * Note these are not specifically called out as "semantic conventions" in the Vercel SDK documentation.
 * However, they are specified as what the collect and set on spans.
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
  result: "result",
} as const;

const VercelUsagePostfixes = {
  completionTokens: "completionTokens",
  promptTokens: "promptTokens",
} as const;

const VercelResultPostfixes = {
  text: "text",
  toolCalls: "toolCalls",
  object: "object",
} as const;

const VercelPromptPostfixes = {
  messages: "messages",
} as const;

const VercelToolCallPostfixes = {
  name: "name",
  args: "args",
  output: "output",
} as const;

const SETTINGS = `${AI_PREFIX}.${AIPrefixes.settings}` as const;

const MODEL_ID = `${AI_PREFIX}.${AIPrefixes.model}.id` as const;

const METADATA = `${AI_PREFIX}.${AIPrefixes.telemetry}.metadata` as const;

const TOKEN_COUNT_COMPLETION =
  `${AI_PREFIX}.${AIPrefixes.usage}.${VercelUsagePostfixes.completionTokens}` as const;

const TOKEN_COUNT_PROMPT =
  `${AI_PREFIX}.${AIPrefixes.usage}.${VercelUsagePostfixes.promptTokens}` as const;

const RESULT_TEXT =
  `${AI_PREFIX}.${AIPrefixes.result}.${VercelResultPostfixes.text}` as const;

const RESULT_TOOL_CALLS =
  `${AI_PREFIX}.${AIPrefixes.result}.${VercelResultPostfixes.toolCalls}` as const;

const RESULT_OBJECT =
  `${AI_PREFIX}.${AIPrefixes.result}.${VercelResultPostfixes.object}` as const;

const PROMPT = `${AI_PREFIX}.${AIPrefixes.prompt}` as const;

const PROMPT_MESSAGES = `${PROMPT}.${VercelPromptPostfixes.messages}` as const;

const EMBEDDING_TEXT = `${AI_PREFIX}.value` as const;
const EMBEDDING_VECTOR = `${AI_PREFIX}.embedding` as const;

const EMBEDDING_TEXTS = `${AI_PREFIX}.values` as const;
const EMBEDDING_VECTORS = `${AI_PREFIX}.embeddings` as const;

const TOOL_CALL_NAME =
  `${AI_PREFIX}.${AIPrefixes.toolCall}.${VercelToolCallPostfixes.name}` as const;
const TOOL_CALL_ARGS =
  `${AI_PREFIX}.${AIPrefixes.toolCall}.${VercelToolCallPostfixes.args}` as const;

/**
 * The semantic conventions used by the Vercel AI SDK.
 * @see https://sdk.vercel.ai/docs/ai-sdk-core/telemetry#collected-data
 */
export const AISemanticConventions = {
  MODEL_ID,
  METADATA,
  SETTINGS,
  TOKEN_COUNT_COMPLETION,
  TOKEN_COUNT_PROMPT,
  RESULT_TEXT,
  RESULT_TOOL_CALLS,
  RESULT_OBJECT,
  PROMPT,
  PROMPT_MESSAGES,
  EMBEDDING_TEXT,
  EMBEDDING_VECTOR,
  EMBEDDING_TEXTS,
  EMBEDDING_VECTORS,
  TOOL_CALL_NAME,
  TOOL_CALL_ARGS,
} as const;

export const AISemanticConventionsList = Object.freeze(
  Object.values(AISemanticConventions),
);

export type AISemanticConvention = ValueOf<typeof AISemanticConventions>;