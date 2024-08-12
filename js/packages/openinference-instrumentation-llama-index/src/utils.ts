import * as llamaindex from "llamaindex";

import { safeExecuteInTheMiddle } from "@opentelemetry/instrumentation";
import {
  Attributes,
  Span,
  SpanKind,
  SpanStatusCode,
  context,
  trace,
  Tracer,
  diag,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import {
  GenericFunction,
  SafeFunction,
  ObjectWithModel,
  RetrieverQueryEngineQueryMethodType,
  RetrieverRetrieveMethodType,
  QueryEmbeddingMethodType,
  LLMChatMethodType,
  LLMCompleteMethodType,
  LLMObject,
} from "./types";
import { BaseEmbedding, BaseRetriever, LLM } from "llamaindex";

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
function getExecContext(span: Span) {
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
function handleError(span: Span, error: Error | undefined) {
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
 * @param {unknown} proto - The prototype to check.
 * @returns {boolean} Whether the prototype is a `BaseRetriever`.
 */
export function isRetrieverPrototype(proto: unknown): proto is BaseRetriever {
  return (
    proto != null &&
    typeof proto === "object" &&
    "retrieve" in proto &&
    typeof proto.retrieve === "function"
  );
}

/**
 * Checks whether the provided prototype is an instance of `LLM`.
 *
 * @param {unknown} proto - The prototype to check.
 * @returns {boolean} Whether the prototype is a `LLM`.
 */
export function isLLMPrototype(proto: unknown): proto is LLM {
  return (
    proto != null &&
    typeof proto === "object" &&
    "chat" in proto &&
    "complete" in proto &&
    typeof proto.chat === "function" &&
    typeof proto.complete === "function"
  );
}

/**
 * Checks whether the provided prototype is an instance of a `BaseEmbedding`.
 *
 * @param {unknown} proto - The prototype to check.
 * @returns {boolean} Whether the prototype is a `BaseEmbedding`.
 */
export function isEmbeddingPrototype(proto: unknown): proto is BaseEmbedding {
  return proto != null && proto instanceof BaseEmbedding;
}

/**
 * Extracts document attributes from an array of nodes with scores and returns extracted
 * attributes in an Attributes object.
 *
 * @param {llamaindex.NodeWithScore<llamaindex.Metadata>[]} output - Array of nodes.
 * @returns {Attributes} The extracted document attributes.
 */
function getDocumentAttributes(
  output: llamaindex.NodeWithScore<llamaindex.Metadata>[],
) {
  const docs: Attributes = {};
  output.forEach(({ node, score }, idx) => {
    if (node instanceof llamaindex.TextNode) {
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
 * @param {string} embeddingInfo.input - The input text for the embedding.
 * @param {number[]} embeddingInfo.output - The output embedding vector.
 * @returns {Attributes} The constructed embedding attributes.
 */
function getQueryEmbeddingAttributes(embeddingInfo: {
  input: string;
  output: number[];
}): Attributes {
  return {
    [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`]:
      embeddingInfo.input,
    [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`]:
      embeddingInfo.output,
  };
}

function getLLMChatAttributes(chatInfo: {
  input: llamaindex.LLMChatParamsNonStreaming<object, object>;
  output: llamaindex.ChatResponse<object>;
}) {
  const LLMAttr: Attributes = {};

  chatInfo.input.messages.forEach((msg, idx) => {
    const inputPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${idx}`;
    LLMAttr[`${inputPrefix}.${SemanticConventions.MESSAGE_ROLE}`] =
      msg.role.toString();
    LLMAttr[`${inputPrefix}.${SemanticConventions.MESSAGE_CONTENT}`] =
      msg.content.toString();
  });

  LLMAttr[SemanticConventions.LLM_INVOCATION_PARAMETERS] =
    safelyJSONStringify(chatInfo.input.additionalChatOptions) ?? undefined;

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
  input: llamaindex.LLMCompletionParamsNonStreaming;
  output: llamaindex.CompletionResponse;
}) {
  const LLMAttr: Attributes = {};

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

function parseLLMMetadata(cls: unknown) {
  const LLMObjectMetadata = (cls as LLMObject).metadata;
  const LLMMetadataAttr: Attributes = {};

  LLMMetadataAttr[SemanticConventions.LLM_MODEL_NAME] = LLMObjectMetadata.model;
  // TODO: How to prevent conflicts with input.additionalChatOptions?
  LLMMetadataAttr[SemanticConventions.LLM_INVOCATION_PARAMETERS] =
    safelyJSONStringify(LLMObjectMetadata) ?? undefined;

  return LLMMetadataAttr;
}

/**
 * Checks if the provided class has a `model` property of type string
 * as a class property.
 *
 * @param {unknown} cls - The class to check.
 * @returns {boolean} Whether the object has a `model` property.
 */
function hasModelProperty(cls: unknown): cls is ObjectWithModel {
  const objectWithModelMaybe = cls as ObjectWithModel;
  return (
    "model" in objectWithModelMaybe &&
    typeof objectWithModelMaybe.model === "string"
  );
}

/**
 * Retrieves the value of the `model` property if the provided class
 * implements it; otherwise, returns undefined.
 *
 * @param {unknown} cls - The class to retrieve the model name from.
 * @returns {string | undefined} The model name or undefined.
 */
function getModelName(cls: unknown) {
  if (hasModelProperty(cls)) {
    return cls.model;
  }
}

export function patchQueryEngineQueryMethod(
  original: RetrieverQueryEngineQueryMethodType,
  tracer: Tracer,
) {
  return function (
    this: unknown,
    ...args: Parameters<RetrieverQueryEngineQueryMethodType>
  ) {
    const span = tracer.startSpan(`query`, {
      kind: SpanKind.INTERNAL,
      attributes: {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.CHAIN,
        [SemanticConventions.INPUT_VALUE]: args[0].query,
        [SemanticConventions.INPUT_MIME_TYPE]: MimeType.TEXT,
      },
    });

    const execContext = getExecContext(span);

    const execPromise = safeExecuteInTheMiddle<
      ReturnType<RetrieverQueryEngineQueryMethodType>
    >(
      () => {
        return context.with(execContext, () => {
          return original.apply(this, args);
        });
      },
      (error) => handleError(span, error),
    );

    const wrappedPromise = execPromise.then((result) => {
      span.setAttributes({
        [SemanticConventions.OUTPUT_VALUE]: result.response,
        [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
      });
      span.end();
      return result;
    });
    return context.bind(execContext, wrappedPromise);
  };
}

export function patchRetrieveMethod(
  original: RetrieverRetrieveMethodType,
  tracer: Tracer,
) {
  return function (
    this: unknown,
    ...args: Parameters<RetrieverRetrieveMethodType>
  ) {
    const span = tracer.startSpan(`retrieve`, {
      kind: SpanKind.INTERNAL,
      attributes: {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.RETRIEVER,
        [SemanticConventions.INPUT_VALUE]: args[0].query,
        [SemanticConventions.INPUT_MIME_TYPE]: MimeType.TEXT,
      },
    });

    const execContext = getExecContext(span);

    const execPromise = safeExecuteInTheMiddle<
      ReturnType<RetrieverRetrieveMethodType>
    >(
      () => {
        return context.with(execContext, () => {
          return original.apply(this, args);
        });
      },
      (error) => handleError(span, error),
    );

    const wrappedPromise = execPromise.then((result) => {
      span.setAttributes(getDocumentAttributes(result));
      span.end();
      return result;
    });
    return context.bind(execContext, wrappedPromise);
  };
}

export function patchQueryEmbeddingMethod(
  original: QueryEmbeddingMethodType,
  tracer: Tracer,
) {
  return function (
    this: unknown,
    ...args: Parameters<QueryEmbeddingMethodType>
  ) {
    const span = tracer.startSpan(`embedding`, {
      kind: SpanKind.INTERNAL,
      attributes: {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.EMBEDDING,
      },
    });

    const execContext = getExecContext(span);

    const execPromise = safeExecuteInTheMiddle<
      ReturnType<QueryEmbeddingMethodType>
    >(
      () => {
        return context.with(execContext, () => {
          return original.apply(this, args);
        });
      },
      (error) => handleError(span, error),
    );

    // Model ID/name is a property found on the class and not in args
    // Extract from class and set as attribute
    span.setAttributes({
      [SemanticConventions.EMBEDDING_MODEL_NAME]: getModelName(this),
    });

    const wrappedPromise = execPromise.then((result) => {
      const [query] = args;
      span.setAttributes(
        getQueryEmbeddingAttributes({ input: query, output: result }),
      );
      span.end();
      return result;
    });
    return context.bind(execContext, wrappedPromise);
  };
}

export function patchLLMChat(original: LLMChatMethodType, tracer: Tracer) {
  return function (this: unknown, ...args: Parameters<LLMChatMethodType>) {
    const span = tracer.startSpan(`llm`, {
      kind: SpanKind.INTERNAL,
      attributes: {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.LLM,
      },
    });

    const execContext = getExecContext(span);

    const execPromise = safeExecuteInTheMiddle<ReturnType<LLMChatMethodType>>(
      () => {
        return context.with(execContext, () => {
          return original.apply(this, args);
        });
      },
      (error) => handleError(span, error),
    );

    // Metadata is a property found on the class and not in args
    span.setAttributes(parseLLMMetadata(this));

    const wrappedPromise = execPromise.then((result) => {
      span.setAttributes(
        getLLMChatAttributes({ input: args[0], output: result }),
      );
      span.end();
      return result;
    });
    return context.bind(execContext, wrappedPromise);
  };
}

export function patchLLMComplete(
  original: LLMCompleteMethodType,
  tracer: Tracer,
) {
  return function (this: unknown, ...args: Parameters<LLMCompleteMethodType>) {
    const span = tracer.startSpan(`llm`, {
      kind: SpanKind.INTERNAL,
      attributes: {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.LLM,
      },
    });

    const execContext = getExecContext(span);

    const execPromise = safeExecuteInTheMiddle<
      ReturnType<LLMCompleteMethodType>
    >(
      () => {
        return context.with(execContext, () => {
          return original.apply(this, args);
        });
      },
      (error) => handleError(span, error),
    );

    // Metadata is a property found on the class and not in args
    span.setAttributes(parseLLMMetadata(this));

    const wrappedPromise = execPromise.then((result) => {
      span.setAttributes(
        getLLMCompleteAttributes({ input: args[0], output: result }),
      );
      span.end();
      return result;
    });
    return context.bind(execContext, wrappedPromise);
  };
}
