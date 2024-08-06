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
  ObjectWithID,
  QueryEngineQueryMethod,
  RetrieverRetrieveMethod,
  QueryEmbeddingMethod,
  TextEmbeddingsMethod,
  LLMChatMethodType,
} from "./types";

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

function queryEmbeddingAttributes(input: string, output: number[]) {
  return {
    [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`]:
      input,
    [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`]:
      output,
  };
}

function textEmbeddingsAttributes(input: string[], output: number[][]) {
  const embedAttr: Attributes = {};
  input.forEach((str, idx) => {
    embedAttr[
      `${SemanticConventions.EMBEDDING_EMBEDDINGS}.${idx}.${SemanticConventions.EMBEDDING_TEXT}`
    ] = str;
  });

  output.forEach((vector, idx) => {
    embedAttr[
      `${SemanticConventions.EMBEDDING_EMBEDDINGS}.${idx}.${SemanticConventions.EMBEDDING_VECTOR}`
    ] = vector;
  });
  return embedAttr;
}

function hasId(obj: unknown): obj is ObjectWithID {
  const objectWithIDMaybe = obj as ObjectWithID;
  return "id" in objectWithIDMaybe && typeof objectWithIDMaybe.id === "string";
}

function hasModel(obj: unknown): obj is ObjectWithModel {
  const ObjectWithModelMaybe = obj as ObjectWithModel;
  return (
    "model" in ObjectWithModelMaybe &&
    typeof ObjectWithModelMaybe.model === "string"
  );
}

function getModelName(obj: unknown) {
  if (hasModel(obj)) {
    return obj.model;
  } else if (hasId(obj)) {
    return obj.id;
  }
}

export function patchQueryMethod(
  original: QueryEngineQueryMethod,
  module: typeof llamaindex,
  tracer: Tracer,
) {
  return function patchedQuery(
    this: unknown,
    ...args: Parameters<QueryEngineQueryMethod>
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
      ReturnType<QueryEngineQueryMethod>
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
  original: RetrieverRetrieveMethod,
  module: typeof llamaindex,
  tracer: Tracer,
) {
  return function patchedRetrieve(
    this: unknown,
    ...args: Parameters<RetrieverRetrieveMethod>
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
      ReturnType<RetrieverRetrieveMethod>
    >(
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

export function patchQueryEmbedding(
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

    // Model ID/name is a property found on the class and not in args
    // Extract from class and set as attribute
    span.setAttributes({
      [SemanticConventions.EMBEDDING_MODEL_NAME]: getModelName(this),
    });

    const wrappedPromise = execPromise.then((result) => {
      span.setAttributes(queryEmbeddingAttributes(args[0], result));
      span.end();
      return result;
    });
    return context.bind(execContext, wrappedPromise);
  };
}

export function patchTextEmbeddings(
  original: TextEmbeddingsMethod,
  tracer: Tracer,
) {
  return function patched(
    this: unknown,
    ...args: Parameters<TextEmbeddingsMethod>
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
      ReturnType<TextEmbeddingsMethod>
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
      span.setAttributes(textEmbeddingsAttributes(args[0], result));
      span.end();
      return result;
    });
    return context.bind(execContext, wrappedPromise);
  };
}

export function patchLLMChat(original: LLMChatMethodType, tracer: Tracer) {
  return function patchedQueryEmbedding(
    this: unknown,
    ...args: Parameters<LLMChatMethodType>
  ) {
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

    args[0].messages.forEach((msg, idx) => {
      span.setAttributes({
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.${idx}.role`]:
          msg.role.toString(),
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.${idx}.role`]:
          msg.content.toString(),
      });
    });

    const wrappedPromise = execPromise.then((result) => {
      // result
      // span.setAttributes({
      //   [SemanticConventions.OUTPUT_VALUE]: result.message.content,
      //   [SemanticConventions.LLM_OUTPUT_MESSAGES]:
      // });
      // const output: Attributes = {}
      // result.message.content
      // result.message.role
      // result.message.options
      span.end();
      return result;
    });
    return context.bind(execContext, wrappedPromise);
  };
}
