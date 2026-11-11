import type { LangSmithParams } from "@langchain/core/language_models/chat_models";

import { LLMProvider, LLMSystem } from "@arizeai/openinference-semantic-conventions";

import { isObject, isString } from "./typeUtils";

/**
 * The LangSmith params that LangChain chat models merge into their run's
 * metadata (see `getLsParams` in `@langchain/core`), narrowed to the ones that
 * map to an OpenInference attribute. The others (`ls_model_type`,
 * `ls_temperature`, `ls_max_tokens` and `ls_stop`) restate the invocation
 * params, which are already recorded as `llm.invocation_parameters`.
 */
export type ParsedLangSmithParams = Pick<LangSmithParams, "ls_provider" | "ls_model_name">;

/**
 * Maps the lowercased `ls_provider` values that LangChain integrations report to
 * OpenInference providers. Mirrors the Python instrumentation's
 * `_LANGCHAIN_PROVIDER_MAP`, plus the names that only the JS integrations use
 * (`cerebras`, `mistral` and `watsonx`). Unlisted providers are recorded as
 * their lowercased `ls_provider`.
 */
const LANGCHAIN_PROVIDER_TO_LLM_PROVIDER: Partial<Record<string, string>> = {
  amazon_bedrock: LLMProvider.AWS,
  anthropic: LLMProvider.ANTHROPIC,
  "anthropic-bedrock": LLMProvider.AWS,
  "anthropic-mantle": LLMProvider.AWS,
  azure: LLMProvider.AZURE,
  azure_ai: LLMProvider.AZURE,
  azure_openai: LLMProvider.AZURE,
  bedrock: LLMProvider.AWS,
  bedrock_converse: LLMProvider.AWS,
  cerebras: LLMProvider.CEREBRAS,
  cohere: LLMProvider.COHERE,
  deepseek: LLMProvider.DEEPSEEK,
  fireworks: LLMProvider.FIREWORKS,
  google: LLMProvider.GOOGLE,
  google_anthropic_vertex: LLMProvider.GOOGLE,
  google_genai: LLMProvider.GOOGLE,
  google_vertexai: LLMProvider.GOOGLE,
  groq: LLMProvider.GROQ,
  huggingface: "huggingface",
  ibm: "ibm",
  mistral: LLMProvider.MISTRALAI,
  mistralai: LLMProvider.MISTRALAI,
  nvidia: "nvidia",
  ollama: LLMProvider.OLLAMA,
  openai: LLMProvider.OPENAI,
  "openai-mantle": LLMProvider.AWS,
  perplexity: LLMProvider.PERPLEXITY,
  together: LLMProvider.TOGETHER,
  vertex: LLMProvider.GOOGLE,
  vertexai: LLMProvider.GOOGLE,
  watsonx: "ibm",
  xai: LLMProvider.XAI,
};

/**
 * Maps the lowercased `ls_provider` values to OpenInference systems (the API
 * shape a provider serves). Mirrors the Python instrumentation's
 * `_PROVIDER_TO_SYSTEM`, plus the JS-only `mistral`. Providers without a
 * matching system are left out, so no `llm.system` is recorded for them.
 */
const LANGCHAIN_PROVIDER_TO_LLM_SYSTEM: Partial<Record<string, LLMSystem>> = {
  anthropic: LLMSystem.ANTHROPIC,
  azure: LLMSystem.OPENAI,
  azure_ai: LLMSystem.OPENAI,
  azure_openai: LLMSystem.OPENAI,
  cohere: LLMSystem.COHERE,
  google: LLMSystem.VERTEXAI,
  google_anthropic_vertex: LLMSystem.ANTHROPIC,
  google_genai: LLMSystem.VERTEXAI,
  google_vertexai: LLMSystem.VERTEXAI,
  groq: LLMSystem.OPENAI,
  mistral: LLMSystem.MISTRALAI,
  mistralai: LLMSystem.MISTRALAI,
  ollama: LLMSystem.OPENAI,
  openai: LLMSystem.OPENAI,
  vertex: LLMSystem.VERTEXAI,
  vertexai: LLMSystem.VERTEXAI,
};

/**
 * Returns the value trimmed if it is a string that isn't blank, otherwise undefined.
 */
export function getNonBlankString(value: unknown): string | undefined {
  if (!isString(value)) {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed === "" ? undefined : trimmed;
}

/**
 * Parses the LangSmith params out of a run's metadata, dropping any that are
 * missing, blank or of the wrong type.
 * @param metadata - The metadata of a langchain run
 * @returns The LangSmith params that are present and well formed
 */
export function parseLangSmithParams(metadata: unknown): ParsedLangSmithParams {
  if (!isObject(metadata)) {
    return {};
  }
  const provider = getNonBlankString(metadata.ls_provider);
  const modelName = getNonBlankString(metadata.ls_model_name);
  return {
    ...(provider != null && { ls_provider: provider }),
    ...(modelName != null && { ls_model_name: modelName }),
  };
}

/**
 * Gets the OpenInference provider for a LangChain `ls_provider`.
 */
export function getLLMProvider(lsProvider: string): string {
  const normalizedProvider = lsProvider.toLowerCase();
  return LANGCHAIN_PROVIDER_TO_LLM_PROVIDER[normalizedProvider] ?? normalizedProvider;
}

/**
 * Gets the OpenInference system for a LangChain `ls_provider`, if there is one.
 */
export function getLLMSystem(lsProvider: string): LLMSystem | undefined {
  return LANGCHAIN_PROVIDER_TO_LLM_SYSTEM[lsProvider.toLowerCase()];
}
