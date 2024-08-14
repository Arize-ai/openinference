/**
 * Below are the semantic conventions for @vercel/otel instrumentation.
 * Note these are not specifically called out as "semantic conventions" in the Vercel SDK documentation.
 * However, they are specified as what the collect and set on spans.
 * @see https://sdk.vercel.ai/docs/ai-sdk-core/telemetry#collected-data
 */
const VERCEL_AI_PREFIX = "ai" as const;

const VercelAIPrefixes = {
  settings: "settings",
  model: "model",
  usage: "usage",
  telemetry: "telemetry",
  prompt: "prompt",
  toolCall: "toolCall",
  result: "result",
} as const;

const VercelUsagePostfixes = {
  completion: "completion",
  prompt: "prompt",
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

const SETTINGS = `${VERCEL_AI_PREFIX}.${VercelAIPrefixes.settings}` as const;

const MODEL_ID = `${VERCEL_AI_PREFIX}.${VercelAIPrefixes.model}.id` as const;

const METADATA =
  `${VERCEL_AI_PREFIX}.${VercelAIPrefixes.telemetry}.metadata` as const;

const TOKEN_COUNT_COMPLETION =
  `${VERCEL_AI_PREFIX}.${VercelAIPrefixes.usage}.${VercelUsagePostfixes.completion}` as const;

const TOKEN_COUNT_PROMPT =
  `${VERCEL_AI_PREFIX}.${VercelAIPrefixes.usage}.${VercelUsagePostfixes.prompt}` as const;

const RESULT_TEXT =
  `${VERCEL_AI_PREFIX}.${VercelAIPrefixes.result}.${VercelResultPostfixes.text}` as const;

const RESULT_TOOL_CALLS =
  `${VERCEL_AI_PREFIX}.${VercelAIPrefixes.result}.${VercelResultPostfixes.toolCalls}` as const;

const RESULT_OBJECT =
  `${VERCEL_AI_PREFIX}.${VercelAIPrefixes.result}.${VercelResultPostfixes.object}` as const;

const PROMPT = `${VERCEL_AI_PREFIX}.${VercelAIPrefixes.prompt}` as const;

const PROMPT_MESSAGES = `${PROMPT}.${VercelPromptPostfixes.messages}` as const;

const VALUE = `${VERCEL_AI_PREFIX}.value` as const;
const EMBEDDING = `${VERCEL_AI_PREFIX}.embedding` as const;

const VALUES = `${VERCEL_AI_PREFIX}.values` as const;
const EMBEDDINGS = `${VERCEL_AI_PREFIX}.embeddings` as const;

const TOOL_CALL_NAME = `${VERCEL_AI_PREFIX}.${VercelAIPrefixes}` as const;
const TOOL_CALL_ARGS =
  `${VERCEL_AI_PREFIX}.${VercelAIPrefixes.toolCall}.${VercelToolCallPostfixes.args}` as const;

export const VercelSemanticConventions = {
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
  VALUE,
  EMBEDDING,
  VALUES,
  EMBEDDINGS,
  TOOL_CALL_NAME,
  TOOL_CALL_ARGS,
} as const;

type ValueOf<T> = T[keyof T];

export type VercelSemanticConvention = ValueOf<
  typeof VercelSemanticConventions
>;
