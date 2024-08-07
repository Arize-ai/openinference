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
 * Checks if
 * @param prototype -
 * @returns
 */
export function isRetriever(
  moduleClassPrototype: unknown,
): moduleClassPrototype is BaseRetriever {
  return (
    !!moduleClassPrototype && !!(moduleClassPrototype as BaseRetriever).retrieve
  );
}

export function isEmbedding(
  moduleClassPrototype: unknown,
): moduleClassPrototype is BaseEmbedding {
  return (
    !!moduleClassPrototype &&
    moduleClassPrototype instanceof BaseEmbedding &&
    !!(moduleClassPrototype as BaseEmbedding).getQueryEmbedding
  );
}

/**
 * Wraps a function with a try-catch block to catch and log any errors.
 * @param fn - A function to wrap with a try-catch block.
 * @returns A function that returns null if an error is thrown.
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
 * Resolves the execution context for the current span
 * If tracing is suppressed, the span is dropped and the current context is returned
 * @param span
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

function getQueryEmbeddingAttributes(embedInfo: {
  input: string;
  output: number[];
}) {
  return {
    [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`]:
      embedInfo.input,
    [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`]:
      embedInfo.output,
  };
}

function hasModel(moduleClass: unknown): moduleClass is ObjectWithModel {
  const ObjectWithModelMaybe = moduleClass as ObjectWithModel;
  return (
    "model" in ObjectWithModelMaybe &&
    typeof ObjectWithModelMaybe.model === "string"
  );
}

function getModelName(moduleClass: unknown) {
  if (hasModel(moduleClass)) {
    return moduleClass.model;
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
