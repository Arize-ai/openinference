import type { Attributes } from "@opentelemetry/api";

import {
  INPUT_MIME_TYPE,
  INPUT_VALUE,
  MimeType,
  OUTPUT_MIME_TYPE,
  OUTPUT_VALUE,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { safelyJSONStringify } from "../utils";
import type {
  Annotation,
  AnnotationScope,
  DecisionTokenCount,
  Document,
  Embedding,
  InputToAttributesFn,
  Message,
  OutputToAttributesFn,
  SpanInput,
  SpanOutput,
  TokenCount,
  Tool,
} from "./types";

/**
 * Converts function arguments into a standardized SpanInput format for tracing.
 *
 * This function processes variable arguments and determines the appropriate MIME type
 * based on the input. Single string arguments are preserved as text, while other
 * types are JSON stringified.
 *
 * @param args - The function arguments to process into span input format
 * @returns A SpanInput object with value and mimeType, or undefined if no args provided
 *
 * @example
 * ```typescript
 * // String input
 * toInputType("hello") // { value: "hello", mimeType: MimeType.TEXT }
 *
 * // Object input
 * toInputType({ key: "value" }) // { value: '{"key":"value"}', mimeType: MimeType.JSON }
 *
 * // Multiple arguments
 * toInputType("arg1", 42) // { value: '["arg1",42]', mimeType: MimeType.JSON }
 * ```
 */
export function toInputType(...args: unknown[]): SpanInput | undefined {
  if (args.length === 0) {
    return undefined;
  }
  if (args.length === 1) {
    const value = args[0];
    if (value == null) {
      return undefined;
    }
    if (typeof value === "string") {
      return {
        value,
        mimeType: MimeType.TEXT,
      };
    }
    return {
      value: safelyJSONStringify(value) ?? "{}",
      mimeType: MimeType.JSON,
    };
  }
  const value = safelyJSONStringify(args);
  if (value == null) {
    return undefined;
  }
  return {
    value,
    mimeType: MimeType.JSON,
  };
}

/**
 * Converts function output into a standardized SpanOutput format for tracing.
 *
 * This function processes the return value of a traced function and determines
 * the appropriate MIME type. String outputs are preserved as text, while other
 * types are JSON stringified.
 *
 * @param result - The function's return value to process into span output format
 * @returns A SpanOutput object with value and mimeType, or undefined if result is null/undefined
 *
 * @example
 * ```typescript
 * // String output
 * toOutputType("success") // { value: "success", mimeType: MimeType.TEXT }
 *
 * // Object output
 * toOutputType({ status: "ok" }) // { value: '{"status":"ok"}', mimeType: MimeType.JSON }
 *
 * // Null/undefined output
 * toOutputType(null) // undefined
 * ```
 */
export function toOutputType(result: unknown): SpanOutput | undefined {
  if (result == null) {
    return undefined;
  }
  if (typeof result === "string") {
    return {
      value: result,
      mimeType: MimeType.TEXT,
    };
  }
  return {
    value: safelyJSONStringify(result) ?? "{}",
    mimeType: MimeType.JSON,
  };
}

/**
 * Default input processing function that converts function arguments to OpenTelemetry attributes.
 *
 * This is a convenience function that combines `toInputType` and `getInputAttributes`
 * to provide a complete input processing pipeline for tracing decorators.
 *
 * @param args - The function arguments to process
 * @returns OpenTelemetry attributes representing the input
 *
 * @example
 * ```typescript
 * const attrs = defaultProcessInput("hello", { key: "value" });
 * // Returns: { [INPUT_VALUE]: '["hello",{"key":"value"}]', [INPUT_MIME_TYPE]: MimeType.JSON }
 * ```
 */
export const defaultProcessInput: InputToAttributesFn = (...args: unknown[]) =>
  getInputAttributes(toInputType(...args));

/**
 * Default output processing function that converts function results to OpenTelemetry attributes.
 *
 * This is a convenience function that combines `toOutputType` and `getOutputAttributes`
 * to provide a complete output processing pipeline for tracing decorators.
 *
 * @param res - The function's return value to process
 * @returns OpenTelemetry attributes representing the output
 *
 * @example
 * ```typescript
 * const attrs = defaultProcessOutput({ status: "success" });
 * // Returns: { [OUTPUT_VALUE]: '{"status":"success"}', [OUTPUT_MIME_TYPE]: MimeType.JSON }
 * ```
 */
export const defaultProcessOutput: OutputToAttributesFn = (res: unknown) =>
  getOutputAttributes(toOutputType(res));

/**
 * Converts a SpanOutput object or string into OpenTelemetry attributes.
 *
 * This function transforms span output data into the standardized attribute format
 * used by OpenTelemetry, mapping values to the appropriate semantic convention keys.
 *
 * @param output - The SpanOutput object, string, or undefined to convert
 * @returns OpenTelemetry attributes with OUTPUT_VALUE and OUTPUT_MIME_TYPE keys
 *
 * @example
 * ```typescript
 * // SpanOutput object
 * getOutputAttributes({ value: "result", mimeType: MimeType.TEXT })
 * // Returns: { [OUTPUT_VALUE]: "result", [OUTPUT_MIME_TYPE]: MimeType.TEXT }
 *
 * // String input
 * getOutputAttributes("simple string")
 * // Returns: { [OUTPUT_VALUE]: "simple string", [OUTPUT_MIME_TYPE]: MimeType.TEXT }
 *
 * // Undefined input
 * getOutputAttributes(undefined) // Returns: {}
 * ```
 */
export function getOutputAttributes(output: SpanOutput | string | null | undefined): Attributes {
  if (output == null) {
    return {};
  }
  if (typeof output === "string") {
    return {
      [OUTPUT_VALUE]: output,
      [OUTPUT_MIME_TYPE]: MimeType.TEXT,
    };
  }
  return {
    [OUTPUT_VALUE]: output.value,
    [OUTPUT_MIME_TYPE]: output.mimeType,
  };
}

/**
 * Converts a SpanInput object or string into OpenTelemetry attributes.
 *
 * This function transforms span input data into the standardized attribute format
 * used by OpenTelemetry, mapping values to the appropriate semantic convention keys.
 *
 * @param input - The SpanInput object, string, or undefined to convert
 * @returns OpenTelemetry attributes with INPUT_VALUE and INPUT_MIME_TYPE keys
 *
 * @example
 * ```typescript
 * // SpanInput object
 * getOutputAttributes({ value: "query", mimeType: MimeType.TEXT })
 * // Returns: { [INPUT_VALUE]: "query", [INPUT_MIME_TYPE]: MimeType.TEXT }
 *
 * // String input
 * getOutputAttributes("simple string")
 * // Returns: { [INPUT_VALUE]: "simple string", [INPUT_MIME_TYPE]: MimeType.TEXT }
 *
 * // Undefined input
 * getOutputAttributes(undefined) // Returns: {}
 * ```
 */
export function getInputAttributes(input: SpanInput | string | undefined): Attributes {
  if (input == null) {
    return {};
  }
  if (typeof input === "string") {
    return {
      [INPUT_VALUE]: input,
      [INPUT_MIME_TYPE]: MimeType.TEXT,
    };
  }
  return {
    [INPUT_VALUE]: input.value,
    [INPUT_MIME_TYPE]: input.mimeType,
  };
}

/**
 * Generates attributes for embedding operations.
 *
 * Creates OpenTelemetry attributes for embedding-related data including
 * model name and embedding vectors with associated text.
 *
 * @param options - Configuration object for embedding attributes
 * @param options.modelName - The name of the embedding model used
 * @param options.embeddings - Array of embedding objects containing text and vector data
 * @returns OpenTelemetry attributes for embedding operations
 *
 * @example
 * ```typescript
 * const attrs = getEmbeddingAttributes({
 *   modelName: "text-embedding-ada-002",
 *   embeddings: [
 *     { text: "hello world", vector: [0.1, 0.2, 0.3] },
 *     { text: "goodbye", vector: [0.4, 0.5, 0.6] }
 *   ]
 * });
 * ```
 */
export function getEmbeddingAttributes(options: {
  modelName?: string;
  embeddings?: Embedding[];
}): Attributes {
  const { modelName, embeddings } = options;
  const attributes: Attributes = {};

  if (modelName != null) {
    attributes[SemanticConventions.EMBEDDING_MODEL_NAME] = modelName;
  }

  if (Array.isArray(embeddings)) {
    embeddings.forEach((embedding, index) => {
      if (embedding.text != null) {
        attributes[
          `${SemanticConventions.EMBEDDING_EMBEDDINGS}.${index}.${SemanticConventions.EMBEDDING_TEXT}`
        ] = embedding.text;
      }
      if (embedding.vector != null) {
        attributes[
          `${SemanticConventions.EMBEDDING_EMBEDDINGS}.${index}.${SemanticConventions.EMBEDDING_VECTOR}`
        ] = embedding.vector;
      }
    });
  }

  return attributes;
}

/**
 * Generates attributes for retriever operations.
 *
 * Creates OpenTelemetry attributes for document retrieval operations,
 * including document content, IDs, metadata, and scores.
 *
 * @param options - Configuration object for retriever attributes
 * @param options.documents - Array of documents retrieved
 * @returns OpenTelemetry attributes for retriever operations
 *
 * @example
 * ```typescript
 * const attrs = getRetrieverAttributes({
 *   documents: [
 *     { content: "Document 1", id: "doc1", score: 0.95 },
 *     { content: "Document 2", id: "doc2", metadata: { source: "web" } }
 *   ]
 * });
 * ```
 */
export function getRetrieverAttributes(options: { documents: Document[] }): Attributes {
  const { documents } = options;
  const attributes: Attributes = {};

  if (!Array.isArray(documents)) {
    return attributes;
  }

  documents.forEach((document, index) => {
    if (document.content != null) {
      attributes[
        `${SemanticConventions.RETRIEVAL_DOCUMENTS}.${index}.${SemanticConventions.DOCUMENT_CONTENT}`
      ] = document.content;
    }
    if (document.id != null) {
      attributes[
        `${SemanticConventions.RETRIEVAL_DOCUMENTS}.${index}.${SemanticConventions.DOCUMENT_ID}`
      ] = document.id;
    }
    if (document.metadata != null) {
      const key = `${SemanticConventions.RETRIEVAL_DOCUMENTS}.${index}.${SemanticConventions.DOCUMENT_METADATA}`;
      if (typeof document.metadata === "string") {
        attributes[key] = document.metadata;
      } else {
        attributes[key] = safelyJSONStringify(document.metadata) ?? "{}";
      }
    }
    if (document.score != null) {
      attributes[
        `${SemanticConventions.RETRIEVAL_DOCUMENTS}.${index}.${SemanticConventions.DOCUMENT_SCORE}`
      ] = document.score;
    }
  });

  return attributes;
}

/**
 * Generates attributes for document operations.
 *
 * Creates OpenTelemetry attributes for individual document processing,
 * including content, ID, metadata, and score information.
 *
 * @param document - The document to generate attributes for
 * @param documentIndex - The index of the document in a collection
 * @param keyPrefix - The prefix for attribute keys (e.g., "reranker.input_documents")
 * @returns OpenTelemetry attributes for the document
 *
 * @example
 * ```typescript
 * const attrs = getDocumentAttributes(
 *   { content: "Sample document", id: "doc123", score: 0.8 },
 *   0,
 *   "reranker.input_documents"
 * );
 * ```
 */
export function getDocumentAttributes(
  document: Document,
  documentIndex: number,
  keyPrefix: string,
): Attributes {
  const attributes: Attributes = {};

  if (document.content != null) {
    attributes[`${keyPrefix}.${documentIndex}.${SemanticConventions.DOCUMENT_CONTENT}`] =
      document.content;
  }
  if (document.id != null) {
    attributes[`${keyPrefix}.${documentIndex}.${SemanticConventions.DOCUMENT_ID}`] = document.id;
  }
  if (document.metadata != null) {
    const key = `${keyPrefix}.${documentIndex}.${SemanticConventions.DOCUMENT_METADATA}`;
    if (typeof document.metadata === "string") {
      attributes[key] = document.metadata;
    } else {
      attributes[key] = safelyJSONStringify(document.metadata) ?? "{}";
    }
  }
  if (document.score != null) {
    attributes[`${keyPrefix}.${documentIndex}.${SemanticConventions.DOCUMENT_SCORE}`] =
      document.score;
  }

  return attributes;
}

type AnnotationTerminology = "annotation" | "evaluation";

const annotationCollectionPrefixes = {
  annotation: {
    span: SemanticConventions.ANNOTATIONS,
    trace: SemanticConventions.TRACE_ANNOTATIONS,
    session: SemanticConventions.SESSION_ANNOTATIONS,
  },
  evaluation: {
    span: SemanticConventions.EVALUATIONS,
    trace: SemanticConventions.TRACE_EVALUATIONS,
    session: SemanticConventions.SESSION_EVALUATIONS,
  },
} as const;

const annotationFieldNames = {
  annotation: {
    name: SemanticConventions.ANNOTATION_NAME,
    score: SemanticConventions.ANNOTATION_SCORE,
    label: SemanticConventions.ANNOTATION_LABEL,
    explanation: SemanticConventions.ANNOTATION_EXPLANATION,
    annotatorKind: SemanticConventions.ANNOTATION_ANNOTATOR_KIND,
    identifier: SemanticConventions.ANNOTATION_IDENTIFIER,
    metadata: SemanticConventions.ANNOTATION_METADATA,
  },
  evaluation: {
    name: SemanticConventions.EVALUATION_NAME,
    score: SemanticConventions.EVALUATION_SCORE,
    label: SemanticConventions.EVALUATION_LABEL,
    explanation: SemanticConventions.EVALUATION_EXPLANATION,
    annotatorKind: SemanticConventions.EVALUATION_ANNOTATOR_KIND,
    identifier: SemanticConventions.EVALUATION_IDENTIFIER,
    metadata: SemanticConventions.EVALUATION_METADATA,
  },
} as const;

const annotationFields = [
  "name",
  "score",
  "label",
  "explanation",
  "annotatorKind",
  "identifier",
  "metadata",
] as const;

/**
 * Generates flattened attributes for annotations at a span, trace, or session scope.
 */
function getScopedAnnotationAttributes(options: {
  annotations: readonly Annotation[];
  terminology: AnnotationTerminology;
  scope: AnnotationScope;
}): Attributes {
  const { annotations, terminology, scope } = options;
  const collectionPrefix = annotationCollectionPrefixes[terminology]?.[scope];
  const fieldNames = annotationFieldNames[terminology];
  if (collectionPrefix == null || fieldNames == null) {
    throw new TypeError(`Invalid annotation terminology or scope: ${terminology}, ${scope}`);
  }
  if (!Array.isArray(annotations)) {
    throw new TypeError("annotations must be an array of annotation objects");
  }

  const attributes: Attributes = {};
  annotations.forEach((result, index) => {
    if (result == null || typeof result !== "object") {
      throw new TypeError("each annotation must be an object");
    }
    if (typeof result.name !== "string") {
      throw new TypeError("each annotation must have a string name");
    }
    if (result.score == null && result.label == null && result.explanation == null) {
      throw new TypeError("each annotation must have at least one of score, label, or explanation");
    }

    for (const field of annotationFields) {
      let value = result[field];
      if (value == null) {
        continue;
      }
      if (field === "metadata" && typeof value !== "string") {
        value = safelyJSONStringify(value) ?? "{}";
      }
      attributes[`${collectionPrefix}.${index}.${fieldNames[field]}`] = value;
    }
  });
  return attributes;
}

/**
 * Generates annotation attributes for a span, trace, or session.
 *
 * Collection indices are assigned in input order. Metadata objects are JSON-stringified,
 * while metadata strings are preserved as provided.
 *
 * @param options - Annotation values and target scope
 * @param options.annotations - Ordered annotations to flatten
 * @param options.scope - Target scope; defaults to `span`
 * @returns OpenTelemetry attributes using annotation terminology
 *
 * @example
 * ```typescript
 * getAnnotationAttributes({
 *   annotations: [{ name: "correctness", score: 0.95 }],
 *   scope: "trace",
 * });
 * ```
 */
export function getAnnotationAttributes(options: {
  annotations: readonly Annotation[];
  scope?: AnnotationScope;
}): Attributes {
  return getScopedAnnotationAttributes({
    annotations: options.annotations,
    terminology: "annotation",
    scope: options.scope ?? "span",
  });
}

/**
 * Generates annotations using evaluation terminology for a span, trace, or session.
 *
 * Collection indices are assigned in input order. Metadata objects are JSON-stringified,
 * while metadata strings are preserved as provided.
 *
 * @param options - Annotation values and target scope
 * @param options.evaluations - Ordered annotations to encode using evaluation terminology
 * @param options.scope - Target scope; defaults to `span`
 * @returns OpenTelemetry attributes using evaluation terminology
 *
 * @example
 * ```typescript
 * getEvaluationAttributes({
 *   evaluations: [{ name: "correctness", label: "correct" }],
 *   scope: "session",
 * });
 * ```
 */
export function getEvaluationAttributes(options: {
  evaluations: readonly Annotation[];
  scope?: AnnotationScope;
}): Attributes {
  return getScopedAnnotationAttributes({
    annotations: options.evaluations,
    terminology: "evaluation",
    scope: options.scope ?? "span",
  });
}

/**
 * Generates attributes for metadata information.
 *
 * @param metadata - The metadata to include in attributes
 * @returns OpenTelemetry attributes containing the metadata
 *
 * @example
 * ```typescript
 * const attrs = getMetadataAttributes({ version: "1.0", env: "prod" });
 * ```
 */
export function getMetadataAttributes(metadata: Record<string, unknown>): Attributes {
  return {
    [SemanticConventions.METADATA]: safelyJSONStringify(metadata) ?? "{}",
  };
}

/**
 * Generates attributes for tool definitions.
 *
 * Creates OpenTelemetry attributes for tool information including
 * name, description, and parameters schema.
 *
 * @param options - Configuration object for tool attributes
 * @param options.name - The name of the tool
 * @param options.description - Optional description of the tool
 * @param options.parameters - Tool parameters as string or object
 * @returns OpenTelemetry attributes for the tool
 *
 * @example
 * ```typescript
 * const attrs = getToolAttributes({
 *   name: "search_tool",
 *   description: "Search for information",
 *   parameters: { query: { type: "string" } }
 * });
 * ```
 */
export function getToolAttributes(options: {
  name: string;
  description?: string;
  parameters: Record<string, unknown>;
}): Attributes {
  const { name, description, parameters } = options;
  const attributes: Attributes = {
    [SemanticConventions.TOOL_NAME]: name,
  };

  if (description != null) {
    attributes[SemanticConventions.TOOL_DESCRIPTION] = description;
  }

  attributes[SemanticConventions.TOOL_PARAMETERS] = safelyJSONStringify(parameters) ?? "{}";

  return attributes;
}

/**
 * The attribute keys a model identification helper writes to. LLM and
 * DECISION spans share the same model name semantics under different
 * namespaces (`llm.*` vs `decision.*`).
 */
interface ModelNameAttributeKeys {
  modelName: string;
  requestModelName: string;
  responseModelName: string;
}

const LLM_MODEL_NAME_KEYS: ModelNameAttributeKeys = {
  modelName: SemanticConventions.LLM_MODEL_NAME,
  requestModelName: SemanticConventions.LLM_REQUEST_MODEL_NAME,
  responseModelName: SemanticConventions.LLM_RESPONSE_MODEL_NAME,
};

const DECISION_MODEL_NAME_KEYS: ModelNameAttributeKeys = {
  modelName: SemanticConventions.DECISION_MODEL_NAME,
  requestModelName: SemanticConventions.DECISION_REQUEST_MODEL_NAME,
  responseModelName: SemanticConventions.DECISION_RESPONSE_MODEL_NAME,
};

/**
 * Generates the model name attributes for a model call. The `model_name` key
 * is the one consumers display as the primary model, so it mirrors
 * responseModelName ?? requestModelName when not set explicitly, matching how
 * instrumentors populate all three keys. The keys default to the `llm.*`
 * namespace; DECISION spans pass the `decision.*` keys.
 */
function getModelNameAttributes(
  options: {
    modelName?: string;
    requestModelName?: string;
    responseModelName?: string;
  },
  keys: ModelNameAttributeKeys = LLM_MODEL_NAME_KEYS,
): Attributes {
  const attributes: Attributes = {};

  const modelName = options.modelName ?? options.responseModelName ?? options.requestModelName;
  if (modelName != null) {
    attributes[keys.modelName] = modelName;
  }

  if (options.requestModelName != null) {
    attributes[keys.requestModelName] = options.requestModelName;
  }

  if (options.responseModelName != null) {
    attributes[keys.responseModelName] = options.responseModelName;
  }

  return attributes;
}

/**
 * Generates attributes for LLM operations.
 *
 * Creates comprehensive OpenTelemetry attributes for LLM interactions
 * including provider, model, messages, token counts, and tools.
 *
 * @param options - Configuration object for LLM attributes
 * @param options.provider - The LLM provider (e.g., "openai", "anthropic")
 * @param options.system - The LLM system type
 * @param options.modelName - The name of the LLM model
 * @param options.requestModelName - The model requested by the caller, as sent in the request.
 * When modelName is omitted, llm.model_name is mirrored from responseModelName ?? requestModelName
 * @param options.responseModelName - The model that actually generated the response, as reported by the provider
 * @param options.invocationParameters - Parameters used for the LLM invocation
 * @param options.inputMessages - Input messages sent to the LLM
 * @param options.outputMessages - Output messages received from the LLM
 * @param options.tokenCount - Token usage information
 * @param options.tools - Tools available to the LLM
 * @returns OpenTelemetry attributes for LLM operations
 *
 * @example
 * ```typescript
 * const attrs = getLLMAttributes({
 *   provider: "openai",
 *   modelName: "gpt-4",
 *   inputMessages: [{ role: "user", content: "Hello" }],
 *   outputMessages: [{ role: "assistant", content: "Hi there!" }],
 *   tokenCount: { prompt: 10, completion: 5, total: 15 }
 * });
 * ```
 */
export function getLLMAttributes(options: {
  provider?: string;
  system?: string;
  modelName?: string;
  requestModelName?: string;
  responseModelName?: string;
  invocationParameters?: Record<string, unknown>;
  inputMessages?: Message[];
  outputMessages?: Message[];
  tokenCount?: TokenCount;
  tools?: Tool[];
}): Attributes {
  const attributes: Attributes = {};

  // Provider attributes
  if (options.provider != null) {
    attributes[SemanticConventions.LLM_PROVIDER] = options.provider.toLowerCase();
  }

  // System attributes
  if (options.system != null) {
    attributes[SemanticConventions.LLM_SYSTEM] = options.system.toLowerCase();
  }

  Object.assign(attributes, getModelNameAttributes(options));

  // Invocation parameters
  if (options.invocationParameters != null) {
    attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS] =
      safelyJSONStringify(options.invocationParameters) ?? "{}";
  }

  // Input messages
  if (Array.isArray(options.inputMessages)) {
    options.inputMessages.forEach((message, messageIndex) => {
      if (message.role != null) {
        attributes[
          `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_ROLE}`
        ] = message.role;
      }
      if (message.content != null) {
        attributes[
          `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENT}`
        ] = message.content;
      }
      if (Array.isArray(message.contents)) {
        message.contents.forEach((content, contentIndex) => {
          if (content.type != null) {
            attributes[
              `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`
            ] = content.type;
          }
          if (content.type === "text" && content.text != null) {
            attributes[
              `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`
            ] = content.text;
          }
          if (content.type === "image" && content.image?.url != null) {
            attributes[
              `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.${SemanticConventions.MESSAGE_CONTENT_IMAGE}.${SemanticConventions.IMAGE_URL}`
            ] = content.image.url;
          }
          if (content.type === "reasoning") {
            if (content.text != null) {
              attributes[
                `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`
              ] = content.text;
            }
            if (content.signature != null) {
              attributes[
                `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.${SemanticConventions.MESSAGE_CONTENT_SIGNATURE}`
              ] = content.signature;
            }
            if (content.data != null) {
              attributes[
                `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.${SemanticConventions.MESSAGE_CONTENT_DATA}`
              ] = content.data;
            }
            if (content.encryptedContent != null) {
              attributes[
                `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.${SemanticConventions.MESSAGE_CONTENT_ENCRYPTED_CONTENT}`
              ] = content.encryptedContent;
            }
          }
        });
      }
      if (message.toolCallId != null) {
        attributes[
          `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_TOOL_CALL_ID}`
        ] = message.toolCallId;
      }
      if (Array.isArray(message.toolCalls)) {
        message.toolCalls.forEach((toolCall, toolCallIndex) => {
          if (toolCall.id != null) {
            attributes[
              `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_ID}`
            ] = toolCall.id;
          }
          if (toolCall.function?.name != null) {
            attributes[
              `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
            ] = toolCall.function.name;
          }
          if (toolCall.function?.arguments != null) {
            const argsJson =
              typeof toolCall.function.arguments === "string"
                ? toolCall.function.arguments
                : (safelyJSONStringify(toolCall.function.arguments) ?? "{}");
            attributes[
              `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
            ] = argsJson;
          }
          if (toolCall.reasoningSignature != null) {
            attributes[
              `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_REASONING_SIGNATURE}`
            ] = toolCall.reasoningSignature;
          }
        });
      }
    });
  }

  // Output messages
  if (Array.isArray(options.outputMessages)) {
    options.outputMessages.forEach((message, messageIndex) => {
      if (message.role != null) {
        attributes[
          `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_ROLE}`
        ] = message.role;
      }
      if (message.content != null) {
        attributes[
          `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENT}`
        ] = message.content;
      }
      if (Array.isArray(message.contents)) {
        message.contents.forEach((content, contentIndex) => {
          if (content.type != null) {
            attributes[
              `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`
            ] = content.type;
          }
          if (content.type === "text" && content.text != null) {
            attributes[
              `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`
            ] = content.text;
          }
          if (content.type === "image" && content.image?.url != null) {
            attributes[
              `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.${SemanticConventions.MESSAGE_CONTENT_IMAGE}.${SemanticConventions.IMAGE_URL}`
            ] = content.image.url;
          }
          if (content.type === "reasoning") {
            if (content.text != null) {
              attributes[
                `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`
              ] = content.text;
            }
            if (content.signature != null) {
              attributes[
                `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.${SemanticConventions.MESSAGE_CONTENT_SIGNATURE}`
              ] = content.signature;
            }
            if (content.data != null) {
              attributes[
                `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.${SemanticConventions.MESSAGE_CONTENT_DATA}`
              ] = content.data;
            }
            if (content.encryptedContent != null) {
              attributes[
                `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.${SemanticConventions.MESSAGE_CONTENT_ENCRYPTED_CONTENT}`
              ] = content.encryptedContent;
            }
          }
        });
      }
      if (message.toolCallId != null) {
        attributes[
          `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_TOOL_CALL_ID}`
        ] = message.toolCallId;
      }
      if (Array.isArray(message.toolCalls)) {
        message.toolCalls.forEach((toolCall, toolCallIndex) => {
          if (toolCall.id != null) {
            attributes[
              `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_ID}`
            ] = toolCall.id;
          }
          if (toolCall.function?.name != null) {
            attributes[
              `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
            ] = toolCall.function.name;
          }
          if (toolCall.function?.arguments != null) {
            const argsJson =
              typeof toolCall.function.arguments === "string"
                ? toolCall.function.arguments
                : (safelyJSONStringify(toolCall.function.arguments) ?? "{}");
            attributes[
              `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
            ] = argsJson;
          }
          if (toolCall.reasoningSignature != null) {
            attributes[
              `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_REASONING_SIGNATURE}`
            ] = toolCall.reasoningSignature;
          }
        });
      }
    });
  }

  // Token count
  if (options.tokenCount != null) {
    if (options.tokenCount.prompt != null) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT] = options.tokenCount.prompt;
    }
    if (options.tokenCount.completion != null) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION] = options.tokenCount.completion;
    }
    if (options.tokenCount.total != null) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL] = options.tokenCount.total;
    }
    if (options.tokenCount.promptDetails != null) {
      const details = options.tokenCount.promptDetails;
      if (details.audio != null) {
        attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO] = details.audio;
      }
      if (details.cacheRead != null) {
        attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] =
          details.cacheRead;
      }
      if (details.cacheWrite != null) {
        attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] =
          details.cacheWrite;
      }
    }
  }

  // Tools
  if (Array.isArray(options.tools)) {
    options.tools.forEach((tool, toolIndex) => {
      if (tool.jsonSchema != null) {
        const schemaJson =
          typeof tool.jsonSchema === "string"
            ? tool.jsonSchema
            : (safelyJSONStringify(tool.jsonSchema) ?? "{}");
        attributes[
          `${SemanticConventions.LLM_TOOLS}.${toolIndex}.${SemanticConventions.TOOL_JSON_SCHEMA}`
        ] = schemaJson;
      }
    });
  }

  return attributes;
}

/**
 * Generates attributes for DECISION operations.
 *
 * Creates OpenTelemetry attributes for a call to a decision model: a model
 * that scores or selects among candidate options supplied in the request
 * rather than generating free-form text. Decision spans identify the model
 * under the `decision.*` namespace instead of `llm.*`, with the same
 * semantics as {@link getLLMAttributes}: `system` names the decision API
 * ecosystem the call conforms to, `provider` names who hosts the model, and
 * the model name keys follow the same request/response mirroring rules.
 *
 * Record the raw request and response with `getInputAttributes` and
 * `getOutputAttributes` (or the default processors); decision spans have no
 * input or output messages.
 *
 * @param options - Configuration object for DECISION attributes
 * @param options.provider - Who hosts the decision model (e.g., "typesafe", "openai").
 * See `DecisionProvider` in the semantic conventions for well-known values
 * @param options.system - The decision API ecosystem the call conforms to (e.g., "typesafe"
 * for the System One / Jev API, "openai" for the OpenAI Decisions API). See `DecisionSystem`
 * in the semantic conventions for well-known values
 * @param options.modelName - The name of the decision model (e.g., "jev-1.13.0")
 * @param options.requestModelName - The model requested by the caller, as sent in the request
 * (e.g., the alias "jev-latest"). When modelName is omitted, decision.model_name is mirrored
 * from responseModelName ?? requestModelName
 * @param options.responseModelName - The model that actually produced the decision, as
 * reported by the provider
 * @param options.tokenCount - Token usage: the tokens sent (`input`) and the tokens that make
 * up the typed answers (`output`)
 * @returns OpenTelemetry attributes for DECISION operations
 *
 * @example
 * ```typescript
 * const attrs = getDecisionAttributes({
 *   system: "typesafe",
 *   provider: "typesafe",
 *   requestModelName: "jev-latest",
 *   responseModelName: "jev-1.13.0",
 *   tokenCount: { input: 412, output: 2 },
 * });
 * ```
 */
export function getDecisionAttributes(options: {
  provider?: string;
  system?: string;
  modelName?: string;
  requestModelName?: string;
  responseModelName?: string;
  tokenCount?: DecisionTokenCount;
}): Attributes {
  const attributes: Attributes = {};

  // Provider attributes
  if (options.provider != null) {
    attributes[SemanticConventions.DECISION_PROVIDER] = options.provider.toLowerCase();
  }

  // System attributes
  if (options.system != null) {
    attributes[SemanticConventions.DECISION_SYSTEM] = options.system.toLowerCase();
  }

  Object.assign(attributes, getModelNameAttributes(options, DECISION_MODEL_NAME_KEYS));

  // Token count
  if (options.tokenCount != null) {
    if (options.tokenCount.input != null) {
      attributes[SemanticConventions.DECISION_TOKEN_COUNT_INPUT] = options.tokenCount.input;
    }
    if (options.tokenCount.output != null) {
      attributes[SemanticConventions.DECISION_TOKEN_COUNT_OUTPUT] = options.tokenCount.output;
    }
  }

  return attributes;
}
