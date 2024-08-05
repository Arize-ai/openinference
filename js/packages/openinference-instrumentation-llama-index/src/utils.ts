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

function documentAttributes(
  output: llamaindex.NodeWithScore<llamaindex.Metadata>[],
): Attributes {
  const docs: Attributes = {};
  output.forEach(({ node, score }, index) => {
    if (node instanceof llamaindex.TextNode) {
      const prefix = `${SemanticConventions.RETRIEVAL_DOCUMENTS}.${index}`;
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

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getModelName(this: any) {
  const modelName: Attributes = {};
  if ("id" in this && typeof this.id === "string") {
    return (modelName[SemanticConventions.EMBEDDING_MODEL_NAME] = this.id);
  } else if ("model" in this && typeof this.model === "string") {
    return (modelName[SemanticConventions.EMBEDDING_MODEL_NAME] = this.model);
  }
  return null;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function hasId(obj: any): obj is { id: string } {
  return "id" in obj && typeof obj.id === "string";
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function hasModel(obj: any): obj is { model: string } {
  return "model" in obj && typeof obj.model === "string";
}

// TODO: Building general method
// function instrumentMethod<T extends (...args: any[]) => any>(
//   original: T,
//   tracer: Tracer,
//   spanKind: OpenInferenceSpanKind,
//   kind: string,
//   attributeExtractor: (this: unknown, ...args: Parameters<T>) => Attributes,
// ) {
//   return function (this: unknown, ...args: Parameters<T>) {
//     const span = tracer.startSpan(kind, {
//       kind: SpanKind.INTERNAL,
//       attributes: {
//         [SemanticConventions.OPENINFERENCE_SPAN_KIND]: spanKind,
//       },
//     });

//     const execContext = getExecContext(span);

//     const execPromise = safeExecuteInTheMiddle<ReturnType<T>>(
//       () => context.with(execContext, () => original.apply(this, args)),
//       (error) => handleError(span, error),
//     );

//     const wrappedPromise = execPromise.then((result) => {
//       span.setAttributes(attributeExtractor.call(this, result, ...args));
//       span.end();
//       return result;
//     });

//     return context.bind(execContext, wrappedPromise);
//   };
// }

// export function patchQueryMethodEXPERIMENT(
//   original: typeof module.RetrieverQueryEngine.prototype.query,
//   module: typeof llamaindex,
//   tracer: Tracer,
// ) {
//   return instrumentMethod(
//     original,
//     tracer,
//     OpenInferenceSpanKind.CHAIN,
//     "query",
//     function (this, result, ...args) {
//       const attributes: Attributes = {
//         [SemanticConventions.INPUT_VALUE]: args[0].query,
//         [SemanticConventions.INPUT_MIME_TYPE]: MimeType.TEXT,
//         [SemanticConventions.OUTPUT_VALUE]: result.response,
//         [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
//       };
//       return attributes;
//     },
//   );
// }

type QueryMethod = typeof llamaindex.RetrieverQueryEngine.prototype.query;

export function patchQueryMethod(
  original: QueryMethod,
  module: typeof llamaindex,
  tracer: Tracer,
) {
  return function patchedQuery(
    this: unknown,
    ...args: Parameters<QueryMethod>
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

    const execPromise = safeExecuteInTheMiddle<ReturnType<QueryMethod>>(
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

type RetrieveMethod = typeof llamaindex.VectorIndexRetriever.prototype.retrieve;

export function patchRetrieveMethod(
  original: RetrieveMethod,
  module: typeof llamaindex,
  tracer: Tracer,
) {
  return function patchedRetrieve(
    this: unknown,
    ...args: Parameters<RetrieveMethod>
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

    const execPromise = safeExecuteInTheMiddle<ReturnType<RetrieveMethod>>(
      () => {
        return context.with(execContext, () => {
          return original.apply(this, args);
        });
      },
      (error) => handleError(span, error),
    );

    const wrappedPromise = execPromise.then((result) => {
      span.setAttributes(documentAttributes(result));
      span.end();
      return result;
    });
    return context.bind(execContext, wrappedPromise);
  };
}

type QueryEmbeddingMethod =
  typeof llamaindex.BaseEmbedding.prototype.getQueryEmbedding;

export function patchQueryEmbeddingMethod(
  original: QueryEmbeddingMethod,
  tracer: Tracer,
) {
  return function patchedQueryEmbedding(
    this: unknown,
    ...args: Parameters<QueryEmbeddingMethod>
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
      ReturnType<QueryEmbeddingMethod>
    >(
      () => {
        return context.with(execContext, () => {
          return original.apply(this, args);
        });
      },
      (error) => handleError(span, error),
    );

    // Check id and model properties to set model_name attribute
    // span.setAttributes(getModelName(this));
    if (hasModel(this)) {
      span.setAttributes({
        [SemanticConventions.EMBEDDING_MODEL_NAME]: this.model,
      });
    } else if (hasId(this)) {
      span.setAttributes({
        [SemanticConventions.EMBEDDING_MODEL_NAME]: this.id,
      });
    }

    const wrappedPromise = execPromise.then((result) => {
      const embeddings: Attributes = {};
      embeddings[
        `${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`
      ] = args[0];
      embeddings[
        `${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`
      ] = `<${result.length} dimensional vector>`;
      span.setAttributes(embeddings);

      span.end();
      return result;
    });
    return context.bind(execContext, wrappedPromise);
  };
}

// export function patchTextEmbedding(
//   original: typeof llamaindex.BaseEmbedding.prototype.getTextEmbedding,
//   tracer: Tracer,
//   modelName: string,
// ) {
//   return function (
//     this: unknown,
//     ...args: Parameters<
//       typeof llamaindex.BaseEmbedding.prototype.getTextEmbedding
//     >
//   ) {
//     const span = tracer.startSpan(`embedding`, {
//       kind: SpanKind.INTERNAL,
//       attributes: {
//         [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
//           OpenInferenceSpanKind.EMBEDDING,
//       },
//     });

//     const execContext = getExecContext(span);

//     const execPromise = safeExecuteInTheMiddle<
//       ReturnType<typeof llamaindex.BaseEmbedding.prototype.getTextEmbedding>
//     >(
//       () => {
//         return context.with(execContext, () => {
//           return original.apply(this, args);
//         });
//       },
//       (error) => {
//         // Push the error to the span
//         if (error) {
//           span.recordException(error);
//           span.setStatus({
//             code: SpanStatusCode.ERROR,
//             message: error.message,
//           });
//           span.end();
//         }
//       },
//     );

//     const wrappedPromise = execPromise.then((result) => {
//       span.setAttributes({
//         [SemanticConventions.EMBEDDING_MODEL_NAME]: modelName,
//       });

//       const embeddings: Attributes = {};
//       embeddings[
//         `${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`
//       ] = args[0];
//       embeddings[
//         `${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`
//       ] = `<${result.length} dimensional vector>`;
//       span.setAttributes(embeddings);

//       span.end();
//       return result;
//     });
//     return context.bind(execContext, wrappedPromise);
//   };
// }

// export function patchTextEmbeddings(
//   original: typeof llamaindex.BaseEmbedding.prototype.getTextEmbeddings,
//   tracer: Tracer,
//   modelName: string,
// ) {
//   return function (
//     this: unknown,
//     ...args: Parameters<
//       typeof llamaindex.BaseEmbedding.prototype.getTextEmbeddings
//     >
//   ) {
//     const span = tracer.startSpan(`embedding`, {
//       kind: SpanKind.INTERNAL,
//       attributes: {
//         [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
//           OpenInferenceSpanKind.EMBEDDING,
//       },
//     });

//     const execContext = getExecContext(span);

//     const execPromise = safeExecuteInTheMiddle<
//       ReturnType<typeof llamaindex.BaseEmbedding.prototype.getTextEmbeddings>
//     >(
//       () => {
//         return context.with(execContext, () => {
//           return original.apply(this, args);
//         });
//       },
//       (error) => {
//         // Push the error to the span
//         if (error) {
//           span.recordException(error);
//           span.setStatus({
//             code: SpanStatusCode.ERROR,
//             message: error.message,
//           });
//           span.end();
//         }
//       },
//     );

//     const wrappedPromise = execPromise.then((result) => {
//       span.setAttributes({
//         [SemanticConventions.EMBEDDING_MODEL_NAME]: modelName,
//       });

//       const embeddings: Attributes = {};
//       result.forEach((embedding, index) => {
//         embeddings[
//           `${SemanticConventions.EMBEDDING_EMBEDDINGS}.${index}.${SemanticConventions.EMBEDDING_TEXT}`
//         ] = args[0];
//         embeddings[
//           `${SemanticConventions.EMBEDDING_EMBEDDINGS}.${index}.${SemanticConventions.EMBEDDING_VECTOR}`
//         ] = `<${embedding.length} dimensional vector>`;
//       });
//       span.setAttributes(embeddings);

//       span.end();
//       return result;
//     });
//     return context.bind(execContext, wrappedPromise);
//   };
// }
