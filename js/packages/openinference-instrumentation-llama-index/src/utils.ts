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
} from "./types";
import { BaseEmbedding, BaseRetriever } from "llamaindex";

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
 * @param {unknown} moduleClassPrototype - The prototype to check.
 * @returns {boolean} Whether the prototype is a `BaseRetriever`.
 */
export function isRetriever(cls: unknown): cls is BaseRetriever {
  return cls != null && (cls as BaseRetriever).retrieve != null;
}

/**
 * Checks whether the provided prototype is an instance of a `BaseEmbedding`.
 *
 * @param {unknown} moduleClassPrototype - The prototype to check.
 * @returns {boolean} Whether the prototype is a `BaseEmbedding`.
 */
export function isEmbedding(cls: unknown): cls is BaseEmbedding {
  return cls != null && cls instanceof BaseEmbedding;
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
 * @param {Object} embedInfo - The embedding information.
 * @param {string} embedInfo.input - The input text for the embedding.
 * @param {number[]} embedInfo.output - The output embedding vector.
 * @returns {Attributes} The constructed embedding attributes.
 */
function getQueryEmbeddingAttributes(embedInfo: {
  input: string;
  output: number[];
}): Attributes {
  return {
    [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`]:
      embedInfo.input,
    [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`]:
      embedInfo.output,
  };
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
      span.setAttributes(
        getQueryEmbeddingAttributes({ input: args[0], output: result }),
      );
      span.end();
      return result;
    });
    return context.bind(execContext, wrappedPromise);
  };
}
