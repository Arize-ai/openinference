/**
 * OpenTelemetry GenAI semantic convention names emitted by the
 * OpenInference -> GenAI conversion, pinned to semantic conventions v1.41.1.
 *
 * These are copied, not imported. Every `gen_ai.*` convention is still
 * unstable, so the only source is `@opentelemetry/semantic-conventions/incubating`,
 * which may break in minor releases and which OpenTelemetry asks libraries not
 * to import at runtime. `test/trace/genai/genAIAttributes.test.ts` checks the
 * registry-backed constants below against that package, so a semantic
 * conventions upgrade that renames one fails the test.
 *
 * Only add a constant here when the conversion emits it.
 *
 * The GenAI conventions are now developed in the
 * `open-telemetry/semantic-conventions-genai` repository, which is the source
 * to check for new and renamed attributes.
 * @see https://opentelemetry.io/docs/specs/semconv/gen-ai/
 * @see https://github.com/open-telemetry/semantic-conventions-genai
 */

// ---------------------------------------------------------------------------
// Registry-backed: each key matches an export of the OpenTelemetry package
// (`ATTR_<KEY>` for attribute names, `GEN_AI_<ATTRIBUTE>_VALUE_<KEY>` for
// well-known values).
// ---------------------------------------------------------------------------

/** Attribute names. */
export const GenAIAttributes = {
  GEN_AI_AGENT_NAME: "gen_ai.agent.name",
  GEN_AI_CONVERSATION_ID: "gen_ai.conversation.id",
  GEN_AI_EMBEDDINGS_DIMENSION_COUNT: "gen_ai.embeddings.dimension.count",
  GEN_AI_INPUT_MESSAGES: "gen_ai.input.messages",
  GEN_AI_OPERATION_NAME: "gen_ai.operation.name",
  GEN_AI_OUTPUT_MESSAGES: "gen_ai.output.messages",
  GEN_AI_OUTPUT_TYPE: "gen_ai.output.type",
  GEN_AI_PROVIDER_NAME: "gen_ai.provider.name",
  GEN_AI_REQUEST_CHOICE_COUNT: "gen_ai.request.choice.count",
  GEN_AI_REQUEST_ENCODING_FORMATS: "gen_ai.request.encoding_formats",
  GEN_AI_REQUEST_FREQUENCY_PENALTY: "gen_ai.request.frequency_penalty",
  GEN_AI_REQUEST_MAX_TOKENS: "gen_ai.request.max_tokens",
  GEN_AI_REQUEST_MODEL: "gen_ai.request.model",
  GEN_AI_REQUEST_PRESENCE_PENALTY: "gen_ai.request.presence_penalty",
  GEN_AI_REQUEST_SEED: "gen_ai.request.seed",
  GEN_AI_REQUEST_STOP_SEQUENCES: "gen_ai.request.stop_sequences",
  GEN_AI_REQUEST_STREAM: "gen_ai.request.stream",
  GEN_AI_REQUEST_TEMPERATURE: "gen_ai.request.temperature",
  GEN_AI_REQUEST_TOP_K: "gen_ai.request.top_k",
  GEN_AI_REQUEST_TOP_P: "gen_ai.request.top_p",
  GEN_AI_RESPONSE_FINISH_REASONS: "gen_ai.response.finish_reasons",
  GEN_AI_RESPONSE_ID: "gen_ai.response.id",
  GEN_AI_RESPONSE_MODEL: "gen_ai.response.model",
  GEN_AI_RETRIEVAL_DOCUMENTS: "gen_ai.retrieval.documents",
  GEN_AI_RETRIEVAL_QUERY_TEXT: "gen_ai.retrieval.query.text",
  GEN_AI_TOOL_CALL_ARGUMENTS: "gen_ai.tool.call.arguments",
  GEN_AI_TOOL_CALL_ID: "gen_ai.tool.call.id",
  GEN_AI_TOOL_CALL_RESULT: "gen_ai.tool.call.result",
  GEN_AI_TOOL_DEFINITIONS: "gen_ai.tool.definitions",
  GEN_AI_TOOL_DESCRIPTION: "gen_ai.tool.description",
  GEN_AI_TOOL_NAME: "gen_ai.tool.name",
  GEN_AI_TOOL_TYPE: "gen_ai.tool.type",
  GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS: "gen_ai.usage.cache_creation.input_tokens",
  GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: "gen_ai.usage.cache_read.input_tokens",
  GEN_AI_USAGE_INPUT_TOKENS: "gen_ai.usage.input_tokens",
  GEN_AI_USAGE_OUTPUT_TOKENS: "gen_ai.usage.output_tokens",
} as const;

/** Well-known values of `gen_ai.operation.name`. */
export const GenAIOperationName = {
  CHAT: "chat",
  EMBEDDINGS: "embeddings",
  EXECUTE_TOOL: "execute_tool",
  INVOKE_AGENT: "invoke_agent",
  RETRIEVAL: "retrieval",
  TEXT_COMPLETION: "text_completion",
} as const;

/** Well-known values of `gen_ai.provider.name`. */
export const GenAIProviderName = {
  ANTHROPIC: "anthropic",
  AWS_BEDROCK: "aws.bedrock",
  AZURE_AI_INFERENCE: "azure.ai.inference",
  AZURE_AI_OPENAI: "azure.ai.openai",
  COHERE: "cohere",
  DEEPSEEK: "deepseek",
  GCP_GEN_AI: "gcp.gen_ai",
  GCP_VERTEX_AI: "gcp.vertex_ai",
  GROQ: "groq",
  MISTRAL_AI: "mistral_ai",
  OPENAI: "openai",
  PERPLEXITY: "perplexity",
  X_AI: "x_ai",
} as const;

/** Well-known values of `gen_ai.output.type`. */
export const GenAIOutputType = {
  JSON: "json",
  TEXT: "text",
} as const;

// ---------------------------------------------------------------------------
// Schema-backed: values defined by the GenAI message JSON schemas and attribute
// examples. The OpenTelemetry package exports no constants for these.
// ---------------------------------------------------------------------------

/** Values of `gen_ai.tool.type`. */
export const GenAIToolType = {
  FUNCTION: "function",
} as const;

/** Message roles. */
export const GenAIRole = {
  ASSISTANT: "assistant",
  TOOL: "tool",
  USER: "user",
} as const;

/** Message part `type` values. */
export const GenAIMessagePartType = {
  BLOB: "blob",
  TEXT: "text",
  TOOL_CALL: "tool_call",
  TOOL_CALL_RESPONSE: "tool_call_response",
  URI: "uri",
} as const;

/** Modalities of `blob` and `uri` message parts. */
export const GenAIModality = {
  IMAGE: "image",
} as const;

/** Output message `finish_reason` values. */
export const GenAIFinishReason = {
  CONTENT_FILTER: "content_filter",
  LENGTH: "length",
  STOP: "stop",
  TOOL_CALL: "tool_call",
} as const;
