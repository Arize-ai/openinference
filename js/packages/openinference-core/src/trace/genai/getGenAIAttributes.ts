import type { Attributes, AttributeValue } from "@opentelemetry/api";

import {
  LLMProvider,
  LLMSystem,
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import type { UnknownRecord } from "./conversionUtils";
import {
  coerceBoolean,
  coerceInteger,
  coerceNumber,
  coerceStringArray,
  getIndexedBuckets,
  isRecord,
  parseJSONArray,
  parseJSONIfString,
  parseJSONRecord,
  serializeJSONLikeValue,
  stringifyJSON,
  toNonEmptyString,
  toOptionalString,
  unflattenAttributes,
} from "./conversionUtils";
import {
  GenAIAttributes,
  GenAIOperationName,
  GenAIOutputType,
  GenAIProviderName,
  GenAIToolType,
} from "./genAIAttributes";
import type { GenAIMessage } from "./messages";
import {
  getInputMessages,
  getOutputMessages,
  getResponseFinishReasons,
  LLM_CHOICES,
} from "./messages";
import { getToolDefinitions } from "./tools";

const EMBEDDING_INVOCATION_PARAMETERS = "embedding.invocation_parameters";
const JSON_RESPONSE_FORMATS = ["json", "json_object", "json_schema"];

const OPERATION_NAME_BY_SPAN_KIND: Partial<Record<string, string>> = {
  [OpenInferenceSpanKind.TOOL]: GenAIOperationName.EXECUTE_TOOL,
  [OpenInferenceSpanKind.EMBEDDING]: GenAIOperationName.EMBEDDINGS,
  [OpenInferenceSpanKind.RETRIEVER]: GenAIOperationName.RETRIEVAL,
  [OpenInferenceSpanKind.AGENT]: GenAIOperationName.INVOKE_AGENT,
};

const PROVIDER_NAME_BY_PROVIDER: Partial<Record<string, string>> = {
  [LLMProvider.AWS]: GenAIProviderName.AWS_BEDROCK,
  [LLMProvider.GOOGLE]: GenAIProviderName.GCP_GEN_AI,
  [LLMProvider.OPENAI]: GenAIProviderName.OPENAI,
  [LLMProvider.ANTHROPIC]: GenAIProviderName.ANTHROPIC,
  [LLMProvider.COHERE]: GenAIProviderName.COHERE,
  [LLMProvider.DEEPSEEK]: GenAIProviderName.DEEPSEEK,
  [LLMProvider.GROQ]: GenAIProviderName.GROQ,
  [LLMProvider.PERPLEXITY]: GenAIProviderName.PERPLEXITY,
  [LLMProvider.XAI]: GenAIProviderName.X_AI,
  [LLMProvider.MISTRALAI]: GenAIProviderName.MISTRAL_AI,
};

/** Used only when `llm.provider` is absent */
const PROVIDER_NAME_BY_SYSTEM: Partial<Record<string, string>> = {
  [LLMSystem.VERTEXAI]: GenAIProviderName.GCP_VERTEX_AI,
  [LLMSystem.OPENAI]: GenAIProviderName.OPENAI,
  [LLMSystem.ANTHROPIC]: GenAIProviderName.ANTHROPIC,
  [LLMSystem.COHERE]: GenAIProviderName.COHERE,
  [LLMSystem.MISTRALAI]: GenAIProviderName.MISTRAL_AI,
};

const USAGE_ATTRIBUTE_MAPPINGS = [
  [SemanticConventions.LLM_TOKEN_COUNT_PROMPT, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS],
  [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS],
  [
    SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
    GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ],
  [
    SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
    GenAIAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
  ],
] as const;

type RequestParameterMapping = {
  genAIKey: string;
  /** Invocation parameter names, in order of precedence */
  sourceKeys: string[];
  coerce: (value: unknown) => AttributeValue | undefined;
};

const REQUEST_PARAMETER_MAPPINGS: RequestParameterMapping[] = [
  {
    genAIKey: GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE,
    sourceKeys: ["temperature"],
    coerce: coerceNumber,
  },
  { genAIKey: GenAIAttributes.GEN_AI_REQUEST_TOP_P, sourceKeys: ["top_p"], coerce: coerceNumber },
  { genAIKey: GenAIAttributes.GEN_AI_REQUEST_TOP_K, sourceKeys: ["top_k"], coerce: coerceNumber },
  {
    genAIKey: GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS,
    sourceKeys: ["max_tokens", "max_completion_tokens", "max_output_tokens"],
    coerce: coerceInteger,
  },
  {
    genAIKey: GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY,
    sourceKeys: ["frequency_penalty"],
    coerce: coerceNumber,
  },
  {
    genAIKey: GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY,
    sourceKeys: ["presence_penalty"],
    coerce: coerceNumber,
  },
  { genAIKey: GenAIAttributes.GEN_AI_REQUEST_SEED, sourceKeys: ["seed"], coerce: coerceInteger },
  {
    genAIKey: GenAIAttributes.GEN_AI_REQUEST_STREAM,
    sourceKeys: ["stream"],
    coerce: coerceBoolean,
  },
  {
    genAIKey: GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES,
    sourceKeys: ["stop_sequences", "stop"],
    coerce: coerceStringArray,
  },
  {
    genAIKey: GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT,
    sourceKeys: ["n", "choice_count", "candidate_count"],
    coerce: coerceInteger,
  },
  {
    genAIKey: GenAIAttributes.GEN_AI_REQUEST_ENCODING_FORMATS,
    sourceKeys: ["encoding_formats", "encoding_format"],
    coerce: coerceStringArray,
  },
];

function getSpanKind(attributes: UnknownRecord): string | undefined {
  const spanKind = attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND];
  return typeof spanKind === "string" ? spanKind.toUpperCase() : undefined;
}

function hasCompletionsAttributes(attributes: UnknownRecord): boolean {
  return Object.keys(attributes).some((key) =>
    [SemanticConventions.LLM_PROMPTS, LLM_CHOICES].some(
      (prefix) => key === prefix || key.startsWith(`${prefix}.`),
    ),
  );
}

/**
 * Resolves `gen_ai.operation.name`. An operation name set explicitly on the
 * span wins, since the OpenInference span kind cannot express every operation.
 * Otherwise the operation is derived from the span kind.
 */
function getOperationName(attributes: UnknownRecord): string | undefined {
  const explicitOperationName = toNonEmptyString(attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]);
  if (explicitOperationName) {
    return explicitOperationName;
  }
  const spanKind = getSpanKind(attributes);
  if (spanKind === OpenInferenceSpanKind.LLM) {
    return hasCompletionsAttributes(attributes)
      ? GenAIOperationName.TEXT_COMPLETION
      : GenAIOperationName.CHAT;
  }
  return spanKind === undefined ? undefined : OPERATION_NAME_BY_SPAN_KIND[spanKind];
}

function toLowerCaseString(value: unknown): string | undefined {
  return typeof value === "string" ? value.toLowerCase() : undefined;
}

function getProviderName(attributes: UnknownRecord): string | undefined {
  const provider = toLowerCaseString(attributes[SemanticConventions.LLM_PROVIDER]);
  const system = toLowerCaseString(attributes[SemanticConventions.LLM_SYSTEM]);
  if (provider === LLMProvider.AZURE) {
    return system === LLMSystem.OPENAI
      ? GenAIProviderName.AZURE_AI_OPENAI
      : GenAIProviderName.AZURE_AI_INFERENCE;
  }
  if (provider === LLMProvider.GOOGLE && system === LLMSystem.VERTEXAI) {
    return GenAIProviderName.GCP_VERTEX_AI;
  }
  const mappedProviderName =
    provider === undefined
      ? PROVIDER_NAME_BY_SYSTEM[system ?? ""]
      : PROVIDER_NAME_BY_PROVIDER[provider];
  return mappedProviderName ?? (system || provider || undefined);
}

function getBaseAttributes(attributes: UnknownRecord): Attributes {
  return {
    [GenAIAttributes.GEN_AI_OPERATION_NAME]: getOperationName(attributes),
    [GenAIAttributes.GEN_AI_PROVIDER_NAME]: getProviderName(attributes),
    [GenAIAttributes.GEN_AI_CONVERSATION_ID]: toNonEmptyString(
      attributes[SemanticConventions.SESSION_ID],
    ),
    [GenAIAttributes.GEN_AI_AGENT_NAME]: toNonEmptyString(
      attributes[SemanticConventions.AGENT_NAME],
    ),
  };
}

function getRequestParameterAttributes(invocationParameters: UnknownRecord): Attributes {
  const genAIAttributes: Attributes = {};
  for (const { genAIKey, sourceKeys, coerce } of REQUEST_PARAMETER_MAPPINGS) {
    for (const sourceKey of sourceKeys) {
      const coercedValue = coerce(invocationParameters[sourceKey]);
      const isDefaultChoiceCount =
        genAIKey === GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT && coercedValue === 1;
      if (coercedValue === undefined || isDefaultChoiceCount) {
        continue;
      }
      genAIAttributes[genAIKey] = coercedValue;
      break;
    }
  }
  return genAIAttributes;
}

function getOutputType(invocationParameters: UnknownRecord): string | undefined {
  const responseFormat = invocationParameters.response_format;
  const responseFormatType = isRecord(responseFormat)
    ? toOptionalString(responseFormat.type)
    : toLowerCaseString(responseFormat);
  if (responseFormatType === undefined) {
    return undefined;
  }
  if (JSON_RESPONSE_FORMATS.includes(responseFormatType)) {
    return GenAIOutputType.JSON;
  }
  return responseFormatType === "text" ? GenAIOutputType.TEXT : undefined;
}

function getRequestAttributes(attributes: UnknownRecord): Attributes {
  const llmParameters = parseJSONRecord(attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS]);
  const embeddingParameters = parseJSONRecord(attributes[EMBEDDING_INVOCATION_PARAMETERS]);
  const invocationParameters = { ...embeddingParameters, ...llmParameters };
  // `llm.model_name` mirrors the response model once a response arrives, so the
  // model the caller asked for is read from `llm.request.model_name` first.
  const requestModel =
    toNonEmptyString(attributes[SemanticConventions.LLM_REQUEST_MODEL_NAME]) ||
    toNonEmptyString(attributes[SemanticConventions.LLM_MODEL_NAME]) ||
    toNonEmptyString(attributes[SemanticConventions.EMBEDDING_MODEL_NAME]) ||
    toNonEmptyString(llmParameters.model) ||
    toNonEmptyString(embeddingParameters.model);
  return {
    [GenAIAttributes.GEN_AI_REQUEST_MODEL]: requestModel,
    ...getRequestParameterAttributes(invocationParameters),
    [GenAIAttributes.GEN_AI_OUTPUT_TYPE]: getOutputType(invocationParameters),
  };
}

function getUsageAttributes(attributes: UnknownRecord): Attributes {
  const genAIAttributes: Attributes = {};
  for (const [openInferenceKey, genAIKey] of USAGE_ATTRIBUTE_MAPPINGS) {
    const tokenCount = attributes[openInferenceKey];
    if (typeof tokenCount === "number") {
      genAIAttributes[genAIKey] = tokenCount;
    }
  }
  return genAIAttributes;
}

/**
 * `gen_ai.input.messages` and `gen_ai.output.messages`. These are `any`-typed
 * in the GenAI semantic conventions, so they are JSON encoded on spans.
 */
function getMessageAttributes({
  attributes,
  outputMessages,
}: {
  attributes: UnknownRecord;
  outputMessages: GenAIMessage[];
}): Attributes {
  const inputMessages = getInputMessages(attributes);
  return {
    [GenAIAttributes.GEN_AI_INPUT_MESSAGES]:
      inputMessages.length > 0 ? stringifyJSON(inputMessages) : undefined,
    [GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]:
      outputMessages.length > 0 ? stringifyJSON(outputMessages) : undefined,
  };
}

function getResponseAttributes({
  attributes,
  outputMessages,
}: {
  attributes: UnknownRecord;
  outputMessages: GenAIMessage[];
}): Attributes {
  const genAIAttributes: Attributes = {};
  const finishReasons = getResponseFinishReasons({ attributes, outputMessages });
  if (finishReasons.length > 0) {
    genAIAttributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] = finishReasons;
  }
  const hasJSONResponse =
    getSpanKind(attributes) === OpenInferenceSpanKind.LLM &&
    attributes[SemanticConventions.OUTPUT_MIME_TYPE] === MimeType.JSON;
  const responsePayload = hasJSONResponse
    ? parseJSONRecord(attributes[SemanticConventions.OUTPUT_VALUE])
    : {};
  genAIAttributes[GenAIAttributes.GEN_AI_RESPONSE_ID] = toNonEmptyString(responsePayload.id);
  // An explicit `llm.response.model_name` wins over the model in the payload,
  // which is only available when the output is a JSON provider response.
  genAIAttributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] =
    toNonEmptyString(attributes[SemanticConventions.LLM_RESPONSE_MODEL_NAME]) ||
    toNonEmptyString(responsePayload.model);
  return genAIAttributes;
}

function getToolCallArguments(attributes: UnknownRecord): string | undefined {
  if (SemanticConventions.TOOL_PARAMETERS in attributes) {
    return serializeJSONLikeValue(attributes[SemanticConventions.TOOL_PARAMETERS]);
  }
  const flattenedToolParameters = unflattenAttributes({
    attributes,
    prefix: `${SemanticConventions.TOOL_PARAMETERS}.`,
  });
  return Object.keys(flattenedToolParameters).length > 0
    ? stringifyJSON(flattenedToolParameters)
    : undefined;
}

function getToolAttributes(attributes: UnknownRecord): Attributes {
  const toolDefinitions = getToolDefinitions(attributes);
  const genAIAttributes: Attributes = {
    [GenAIAttributes.GEN_AI_TOOL_DEFINITIONS]:
      toolDefinitions.length > 0 ? stringifyJSON(toolDefinitions) : undefined,
  };
  if (getSpanKind(attributes) !== OpenInferenceSpanKind.TOOL) {
    return genAIAttributes;
  }
  const toolName = toNonEmptyString(attributes[SemanticConventions.TOOL_NAME]);
  const hasToolResult = SemanticConventions.OUTPUT_VALUE in attributes;
  return {
    ...genAIAttributes,
    [GenAIAttributes.GEN_AI_TOOL_NAME]: toolName,
    [GenAIAttributes.GEN_AI_TOOL_TYPE]: toolName ? GenAIToolType.FUNCTION : undefined,
    [GenAIAttributes.GEN_AI_TOOL_DESCRIPTION]: toNonEmptyString(
      attributes[SemanticConventions.TOOL_DESCRIPTION],
    ),
    [GenAIAttributes.GEN_AI_TOOL_CALL_ID]:
      toNonEmptyString(attributes[SemanticConventions.TOOL_ID]) ||
      toNonEmptyString(attributes[SemanticConventions.TOOL_CALL_ID]),
    [GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS]: getToolCallArguments(attributes),
    [GenAIAttributes.GEN_AI_TOOL_CALL_RESULT]: hasToolResult
      ? serializeJSONLikeValue(attributes[SemanticConventions.OUTPUT_VALUE])
      : undefined,
  };
}

/**
 * Reads the first non-nullish value from a document, preferring the
 * OpenInference attribute name over the plain field name.
 */
function getDocumentField({
  document,
  attributeKey,
  fieldName,
}: {
  document: UnknownRecord;
  attributeKey: string;
  fieldName: string;
}): unknown {
  return document[attributeKey] ?? document[fieldName] ?? undefined;
}

function normalizeDocument(document: unknown): UnknownRecord | undefined {
  if (!isRecord(document)) {
    return undefined;
  }
  const documentId = getDocumentField({
    document,
    attributeKey: SemanticConventions.DOCUMENT_ID,
    fieldName: "id",
  });
  const metadata = getDocumentField({
    document,
    attributeKey: SemanticConventions.DOCUMENT_METADATA,
    fieldName: "metadata",
  });
  const normalizedDocument: UnknownRecord = {
    id: documentId === undefined ? undefined : toOptionalString(documentId),
    score: getDocumentField({
      document,
      attributeKey: SemanticConventions.DOCUMENT_SCORE,
      fieldName: "score",
    }),
    content: getDocumentField({
      document,
      attributeKey: SemanticConventions.DOCUMENT_CONTENT,
      fieldName: "content",
    }),
    metadata: metadata === undefined ? undefined : parseJSONIfString(metadata),
  };
  const hasFields = Object.values(normalizedDocument).some((value) => value !== undefined);
  return hasFields ? normalizedDocument : undefined;
}

function getRetrievalDocuments(attributes: UnknownRecord): UnknownRecord[] {
  const rootDocuments = parseJSONArray(attributes[SemanticConventions.RETRIEVAL_DOCUMENTS])
    .map(normalizeDocument)
    .filter((document) => document !== undefined);
  if (rootDocuments.length > 0) {
    return rootDocuments;
  }
  return getIndexedBuckets({ attributes, prefix: SemanticConventions.RETRIEVAL_DOCUMENTS })
    .map(normalizeDocument)
    .filter((document) => document !== undefined);
}

function getRetrievalAttributes(attributes: UnknownRecord): Attributes {
  if (getSpanKind(attributes) !== OpenInferenceSpanKind.RETRIEVER) {
    return {};
  }
  const documents = getRetrievalDocuments(attributes);
  const hasJSONInput = attributes[SemanticConventions.INPUT_MIME_TYPE] === MimeType.JSON;
  return {
    [GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS]:
      documents.length > 0 ? stringifyJSON(documents) : undefined,
    [GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT]: hasJSONInput
      ? undefined
      : toOptionalString(attributes[SemanticConventions.INPUT_VALUE]),
  };
}

function getEmbeddingDimensionCount(attributes: UnknownRecord): number | undefined {
  for (const embedding of parseJSONArray(attributes[SemanticConventions.EMBEDDING_EMBEDDINGS])) {
    if (!isRecord(embedding)) {
      continue;
    }
    const vector = embedding[SemanticConventions.EMBEDDING_VECTOR] || embedding.vector;
    if (Array.isArray(vector)) {
      return vector.length;
    }
  }
  const embeddingPrefix = `${SemanticConventions.EMBEDDING_EMBEDDINGS}.`;
  const vectorSuffix = `.${SemanticConventions.EMBEDDING_VECTOR}`;
  for (const [key, value] of Object.entries(attributes)) {
    if (!key.startsWith(embeddingPrefix) || !key.endsWith(vectorSuffix)) {
      continue;
    }
    const vector = parseJSONIfString(value);
    if (Array.isArray(vector)) {
      return vector.length;
    }
  }
  return undefined;
}

function getEmbeddingAttributes(attributes: UnknownRecord): Attributes {
  if (getSpanKind(attributes) !== OpenInferenceSpanKind.EMBEDDING) {
    return {};
  }
  return {
    [GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT]:
      getEmbeddingDimensionCount(attributes) || undefined,
  };
}

function removeUndefinedValues<Value>(
  record: Record<string, Value | undefined>,
): Record<string, Value> {
  const definedValues: Record<string, Value> = {};
  for (const [key, value] of Object.entries(record)) {
    if (value !== undefined) {
      definedValues[key] = value;
    }
  }
  return definedValues;
}

/**
 * Derives OpenTelemetry GenAI semantic convention (`gen_ai.*`) attributes from
 * a span's OpenInference attributes.
 *
 * This is the conversion applied at span end by {@link OISpan} when
 * `enableGenAISemconv` (`OPENINFERENCE_ENABLE_GENAI_SEMCONV`) is on. It only
 * returns the derived attributes; callers decide how to merge them and must not
 * overwrite `gen_ai.*` attributes that are already on the span.
 *
 * @experimental The GenAI semantic conventions are still in development and the
 * mapping may change.
 * @see https://opentelemetry.io/docs/specs/semconv/gen-ai/
 *
 * @example
 * ```typescript
 * getGenAIAttributes({
 *   "openinference.span.kind": "LLM",
 *   "llm.model_name": "gpt-4o",
 *   "llm.provider": "openai",
 *   "llm.token_count.prompt": 12,
 * });
 * // {
 * //   "gen_ai.operation.name": "chat",
 * //   "gen_ai.provider.name": "openai",
 * //   "gen_ai.request.model": "gpt-4o",
 * //   "gen_ai.usage.input_tokens": 12,
 * // }
 * ```
 *
 * @param attributes - The OpenInference attributes of a span
 * @returns The derived `gen_ai.*` attributes
 */
export function getGenAIAttributes(attributes?: Attributes | null): Attributes {
  if (attributes == null) {
    return {};
  }
  const sourceAttributes: UnknownRecord = removeUndefinedValues(attributes);
  if (Object.keys(sourceAttributes).length === 0) {
    return {};
  }
  const outputMessages = getOutputMessages(sourceAttributes);
  return removeUndefinedValues({
    ...getBaseAttributes(sourceAttributes),
    ...getRequestAttributes(sourceAttributes),
    ...getUsageAttributes(sourceAttributes),
    ...getMessageAttributes({ attributes: sourceAttributes, outputMessages }),
    ...getResponseAttributes({ attributes: sourceAttributes, outputMessages }),
    ...getToolAttributes(sourceAttributes),
    ...getRetrievalAttributes(sourceAttributes),
    ...getEmbeddingAttributes(sourceAttributes),
  });
}
