import { safeExecuteInTheMiddle } from "@opentelemetry/instrumentation";
import {
  SpanKind,
  context,
  Tracer,
  Span,
  Attributes,
} from "@opentelemetry/api";
import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import {
  RetrieverQueryEngineQueryMethodType,
  RetrieverRetrieveMethodType,
  QueryEmbeddingMethodType,
  LLMChatMethodType,
  LLMCompleteMethodType,
} from "./types";
import {
  getExecContext,
  handleError,
  safelyGetDocumentAttributes,
  safelyGetEmbeddingAttributes,
  safelyGetLLMChatAttributes,
  safelyGetLLMCompleteAttributes,
} from "./utils";

function setSpanAttributes(params: {
  span: Span;
  attributes: Attributes | null;
}) {
  if (params.attributes != null) {
    params.span.setAttributes(params.attributes);
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
      setSpanAttributes({
        span: span,
        attributes: safelyGetDocumentAttributes(result),
      });
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

    const wrappedPromise = execPromise.then((result) => {
      // Pass in `this` as model name is a property found on prototype
      setSpanAttributes({
        span: span,
        attributes: safelyGetEmbeddingAttributes({
          prototype: this,
          input: args[0],
          output: result,
        }),
      });
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

    const wrappedPromise = execPromise.then((result) => {
      setSpanAttributes({
        span: span,
        attributes: safelyGetLLMChatAttributes({
          prototype: this,
          input: args[0],
          output: result,
        }),
      });
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

    const wrappedPromise = execPromise.then((result) => {
      setSpanAttributes({
        span: span,
        attributes: safelyGetLLMCompleteAttributes({
          prototype: this,
          input: args[0],
          output: result,
        }),
      });
      span.end();
      return result;
    });
    return context.bind(execContext, wrappedPromise);
  };
}
