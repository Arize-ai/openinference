import type * as llamaindex from "llamaindex";

import { TextNode } from "llamaindex";
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
import { GenericFunction, SafeFunction } from "./types";

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

export function patchQueryMethod(
  original: typeof module.RetrieverQueryEngine.prototype.query,
  module: typeof llamaindex,
  tracer: Tracer,
) {
  return function patchedQuery(
    this: unknown,
    ...args: Parameters<typeof module.RetrieverQueryEngine.prototype.query>
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
      ReturnType<typeof module.RetrieverQueryEngine.prototype.query>
    >(
      () => {
        return context.with(execContext, () => {
          return original.apply(this, args);
        });
      },
      (error) => {
        // Push the error to the span
        if (error) {
          span.recordException(error);
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: error.message,
          });
          span.end();
        }
      },
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
  original: typeof module.VectorIndexRetriever.prototype.retrieve,
  module: typeof llamaindex,
  tracer: Tracer,
) {
  return function patchedRetrieve(
    this: unknown,
    ...args: Parameters<typeof module.VectorIndexRetriever.prototype.retrieve>
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
      ReturnType<typeof module.VectorIndexRetriever.prototype.retrieve>
    >(
      () => {
        return context.with(execContext, () => {
          return original.apply(this, args);
        });
      },
      (error) => {
        // Push the error to the span
        if (error) {
          span.recordException(error);
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: error.message,
          });
          span.end();
        }
      },
    );

    const wrappedPromise = execPromise.then((result) => {
      span.setAttributes(documentAttributes(result));
      span.end();
      return result;
    });
    return context.bind(execContext, wrappedPromise);
  };
}

function documentAttributes(
  output: llamaindex.NodeWithScore<llamaindex.Metadata>[],
) {
  const docs: Attributes = {};
  output.forEach((document, index) => {
    const { node, score } = document;

    if (node instanceof TextNode) {
      const nodeId = node.id_;
      const nodeText = node.getContent();
      const nodeMetadata = node.metadata;

      const prefix = `${SemanticConventions.RETRIEVAL_DOCUMENTS}.${index}`;
      docs[`${prefix}.${SemanticConventions.DOCUMENT_ID}`] = nodeId;
      docs[`${prefix}.${SemanticConventions.DOCUMENT_SCORE}`] = score;
      docs[`${prefix}.${SemanticConventions.DOCUMENT_CONTENT}`] = nodeText;
      docs[`${prefix}.${SemanticConventions.DOCUMENT_METADATA}`] =
        safelyJSONStringify(nodeMetadata) ?? undefined;
    }
  });
  return docs;
}
