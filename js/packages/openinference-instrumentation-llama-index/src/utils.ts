import * as llamaindex from "llamaindex";

import {
  Attributes,
  Span,
  SpanStatusCode,
  context,
  trace,
  diag,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";
import {
  GenericFunction,
  SafeFunction,
  ObjectWithModel,
  ObjectWithMetadata,
} from "./types";
import { BaseEmbedding } from "@llamaindex/core/dist/embeddings";
import {
  LLM,
  LLMChatParamsNonStreaming,
  LLMCompletionParamsNonStreaming,
  ChatResponse,
  CompletionResponse,
  MessageContentDetail,
} from "@llamaindex/core/dist/llms";
import {
  NodeWithScore,
  TextNode,
  Metadata,
} from "@llamaindex/core/dist/schema";

/**
 * Wraps a function with a try-catch block to catch and log any errors.
 * @param {T} fn - A function to wrap with a try-catch block.
 * @returns {SafeFunction<T>} A function that returns null if an error is thrown.
 */
export function withSafety<T extends GenericFunction>(fn: T): SafeFunction<T> {
  return (...args) => {
    try {
      return fn(...args);
    } catch (error) {
      diag.error(`Failed to get attributes for span: ${error}`);
      return null;
    }
  };
}
const safelyJSONStringify = withSafety(JSON.stringify);

/**
 * Resolves the execution context for the current span.
 * If tracing is suppressed, the span is dropped and the current context is returned.
 * @param {Span} span - Tracer span
 * @returns {Context} An execution context.
 */
export function getExecContext(span: Span) {
  const activeContext = context.active();
  const suppressTracing = isTracingSuppressed(activeContext);
  const execContext = suppressTracing
    ? trace.setSpan(context.active(), span)
    : activeContext;
  // Drop the span from the context
  if (suppressTracing) {
    trace.deleteSpan(activeContext);
  }
  return execContext;
}

/**
 * If execution results in an error, push the error to the span.
 * @param span
 * @param error
 */
export function handleError(span: Span, error: Error | undefined) {
  if (error) {
    span.recordException(error);
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: error.message,
    });
    span.end();
  }
}

/**
 * Checks whether the provided prototype is an instance of a `BaseRetriever`.
 *
 * @param {unknown} prototype - The prototype to check.
 * @returns {boolean} Whether the prototype is a `BaseRetriever`.
 */
export function isRetrieverPrototype(
  prototype: unknown,
): prototype is llamaindex.BaseRetriever {
  return (
    prototype != null &&
    typeof prototype === "object" &&
    "retrieve" in prototype &&
    typeof prototype.retrieve === "function"
  );
}

/**
 * Checks whether the provided prototype is an instance of `LLM`.
 *
 * @param {unknown} prototype - The prototype to check.
 * @returns {boolean} Whether the prototype is a `LLM`.
 */
export function isLLMPrototype(prototype: unknown): prototype is LLM {
  return (
    prototype != null &&
    typeof prototype === "object" &&
    "chat" in prototype &&
    "complete" in prototype &&
    typeof prototype.chat === "function" &&
    typeof prototype.complete === "function"
  );
}

/**
 * Checks whether the provided prototype is an instance of a `BaseEmbedding`.
 *
 * @param {unknown} prototype - The prototype to check.
 * @returns {boolean} Whether the prototype is a `BaseEmbedding`.
 */
export function isEmbeddingPrototype(
  prototype: unknown,
): prototype is BaseEmbedding {
  return prototype != null && prototype instanceof BaseEmbedding;
}

/**
 * Checks if the provided class prototype has a `model` property of type string
 * as a class property.
 *
 * @param {unknown} prototype - The prototype to check.
 * @returns {boolean} Whether the object has a `model` property.
 */
function hasModelProperty(prototype: unknown): prototype is ObjectWithModel {
  const objectWithModelMaybe = prototype as ObjectWithModel;
  return (
    "model" in objectWithModelMaybe &&
    typeof objectWithModelMaybe.model === "string"
  );
}

/**
 * Checks if the provided class prototype has a `metadata` property as a class property.
 *
 * @param {unknown} prototype - The class prototype to check.
 * @returns {boolean} Whether the object has a `metadata` property.
 */
function hasMetadataProperty(
  prototype: unknown,
): prototype is ObjectWithMetadata {
  const objectWithMetadataMaybe = prototype as ObjectWithMetadata;
  return (
    "metadata" in objectWithMetadataMaybe &&
    objectWithMetadataMaybe.metadata.contextWindow != null &&
    objectWithMetadataMaybe.metadata.model != null &&
    objectWithMetadataMaybe.metadata.temperature != null &&
    objectWithMetadataMaybe.metadata.tokenizer != null &&
    objectWithMetadataMaybe.metadata.topP != null
  );
}

/**
 * Retrieves the value of the `model` property if the provided class prototype
 * implements it; otherwise, returns undefined.
 *
 * @param {unknown} prototype - The prototype to retrieve the model name from.
 * @returns {string | undefined} The model name or undefined.
 */
function getModelNameProperty(prototype: unknown) {
  if (hasModelProperty(prototype)) {
    return prototype.model;
  }
}

/**
 * Retrieves the value of the `model` property if the provided class prototype
 * implements it; otherwise, returns undefined.
 *
 * @param {unknown} prototype - The prototype to retrieve the model name from.
 * @returns {string | undefined} The model name or undefined.
 */
function getMetadataProperty(prototype: unknown) {
  if (hasMetadataProperty(prototype)) {
    return prototype.metadata;
  }
}

// function getLLMInvocationParams({ stuffThatHasParams1, stuffThatHasParams2 }) {
//   const firstParams = someFunc(stuffThatHasParams1);
//   const otherParams = someFunc2(stuffThatHasParams2);
//   return safelyJSONStringify({ ...firstParams, ...otherParams });
// }

/**
 * Retrieves properties from the class prototype, such as LLM metadata or model name.
 *
 * @param {unknown} prototype - The prototype to retrieve the properties from.
 * @returns {Attributes} Prototype properties set as attributes.
 */
function parseLLMPrototypeProperties(prototype: unknown): Attributes {
  return {
    // TODO: How to prevent conflicts with input.additionalChatOptions?
    [SemanticConventions.LLM_INVOCATION_PARAMETERS]:
      safelyJSONStringify(getMetadataProperty(prototype)) ?? undefined,

    [SemanticConventions.LLM_MODEL_NAME]:
      safelyJSONStringify(getModelNameProperty(prototype)) ?? undefined,
  };
}

/**
 * Extracts document attributes from an array of nodes with scores and returns extracted
 * attributes in an Attributes object.
 *
 * @param {NodeWithScore<Metadata>[]} output - Array of nodes.
 * @returns {Attributes} The extracted document attributes.
 */
function getDocumentAttributes(output: NodeWithScore<Metadata>[]) {
  const docs: Attributes = {};
  output.forEach(({ node, score }, idx) => {
    if (node instanceof TextNode) {
      const prefix = `${SemanticConventions.RETRIEVAL_DOCUMENTS}.${idx}`;
      docs[`${prefix}.${SemanticConventions.DOCUMENT_ID}`] = node.id_;
      docs[`${prefix}.${SemanticConventions.DOCUMENT_SCORE}`] = score;
      docs[`${prefix}.${SemanticConventions.DOCUMENT_CONTENT}`] =
        node.getContent();
      docs[`${prefix}.${SemanticConventions.DOCUMENT_METADATA}`] =
        safelyJSONStringify(node.metadata) ?? undefined;
    }
  });
  return docs;
}

/**
 * Extracts embedding information (input text and the output embedding vector),
 * and constructs an Attributes object with the relevant semantic conventions
 * for embeddings.
 *
 * @param {Object} embeddingInfo - The embedding information.
 * @param {MessageContentDetail} embeddingInfo.input - The input for the embedding.
 * @param {number[]} embeddingInfo.output - The output embedding vector.
 * @returns {Attributes} The constructed embedding attributes.
 */
function getQueryEmbeddingAttributes(embeddingInfo: {
  prototype: unknown;
  input: MessageContentDetail;
  output: number[] | null;
}) {
  const embedAttr: Attributes = {};

  if (embeddingInfo.input.type === "text") {
    embedAttr[
      `${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`
    ] = embeddingInfo.input.text;
  }

  if (embeddingInfo.output) {
    embedAttr[
      `${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`
    ] = embeddingInfo.output;
  }

  // Extract model name class property
  embedAttr[SemanticConventions.EMBEDDING_MODEL_NAME] = getModelNameProperty(
    embeddingInfo.prototype,
  );
  return embedAttr;
}

const safelyGetLLMPrototypeProperties = withSafety(parseLLMPrototypeProperties);

function getLLMChatAttributes(chatInfo: {
  prototype: unknown;
  input: LLMChatParamsNonStreaming<object, object>;
  output: ChatResponse<object>;
}) {
  const LLMAttr: Attributes = {};

  // Extract and set class prototype properties as attributes
  const prototypeProperties = safelyGetLLMPrototypeProperties(
    chatInfo.prototype,
  );
  if (prototypeProperties != null) {
    Object.keys(prototypeProperties).forEach((key) => {
      LLMAttr[key] = prototypeProperties[key];
    });
  }

  chatInfo.input.messages.forEach((msg, idx) => {
    const inputPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${idx}`;
    LLMAttr[`${inputPrefix}.${SemanticConventions.MESSAGE_ROLE}`] =
      msg.role.toString();
    LLMAttr[`${inputPrefix}.${SemanticConventions.MESSAGE_CONTENT}`] =
      msg.content.toString();
  });

  // TODO
  // LLMAttr[SemanticConventions.LLM_INVOCATION_PARAMETERS] =
  //   safelyJSONStringify(chatInfo.input.additionalChatOptions) ?? undefined;

  const outputPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0`;
  LLMAttr[SemanticConventions.OUTPUT_VALUE] =
    chatInfo.output.message.content.toString();
  LLMAttr[`${outputPrefix}.${SemanticConventions.MESSAGE_ROLE}`] =
    chatInfo.output.message.role.toString();
  LLMAttr[`${outputPrefix}.${SemanticConventions.MESSAGE_CONTENT}`] =
    chatInfo.output.message.content.toString();

  return LLMAttr;
}

function getLLMCompleteAttributes(completeInfo: {
  prototype: unknown;
  input: LLMCompletionParamsNonStreaming;
  output: CompletionResponse;
}) {
  const LLMAttr: Attributes = {};

  // Extract and set class prototype properties as attributes
  const prototypeProperties = safelyGetLLMPrototypeProperties(
    completeInfo.prototype,
  );
  if (prototypeProperties != null) {
    Object.keys(prototypeProperties).forEach((key) => {
      LLMAttr[key] = prototypeProperties[key];
    });
  }

  const inputPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.0`;
  LLMAttr[`${inputPrefix}.${SemanticConventions.MESSAGE_CONTENT}`] =
    completeInfo.input.prompt.toString();
  LLMAttr[`${inputPrefix}.${SemanticConventions.MESSAGE_ROLE}`] = "user";

  const outputPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0`;
  LLMAttr[SemanticConventions.OUTPUT_VALUE] = completeInfo.output.text;
  LLMAttr[`${outputPrefix}.${SemanticConventions.MESSAGE_ROLE}`] = "assistant";
  LLMAttr[`${outputPrefix}.${SemanticConventions.MESSAGE_CONTENT}`] =
    completeInfo.output.text;

  return LLMAttr;
}

export const safelyGetDocumentAttributes = withSafety(getDocumentAttributes);
export const safelyGetEmbeddingAttributes = withSafety(
  getQueryEmbeddingAttributes,
);
export const safelyGetLLMChatAttributes = withSafety(getLLMChatAttributes);
export const safelyGetLLMCompleteAttributes = withSafety(
  getLLMCompleteAttributes,
);
