import type * as llamaindex from "llamaindex";

import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
  safeExecuteInTheMiddle,
} from "@opentelemetry/instrumentation";
import { VERSION } from "./version";
import {
  Span,
  SpanKind,
  SpanStatusCode,
  context,
  diag,
  trace,
  Tracer,
} from "@opentelemetry/api";
import { isAttributeValue, isTracingSuppressed } from "@opentelemetry/core";

import {
  ChromaVectorStore,
  Document,
  VectorStoreIndex,
  storageContextFromDefaults,
  TextNode
} from "llamaindex";

const MODULE_NAME = "llamaindex";

/**
 * Flag to check if the LlamaIndex module has been patched
 * Note: This is a fallback in case the module is made immutable (e.x. Deno, webpack, etc.)
 */
let _isOpenInferencePatched = false;

/**
 * function to check if instrumentation is enabled / disabled
 */
export function isPatched() {
  return _isOpenInferencePatched;
}

import {
  OpenInferenceSpanKind,
  SemanticConventions,
  RetrievalAttributePostfixes,
  SemanticAttributePrefixes,
} from "@arizeai/openinference-semantic-conventions";

// TODO: Type imports
import {
  GenericFunction,
  LLMMessage,
  LLMMessageFunctionCall,
  LLMMessageToolCalls,
  LLMMessagesAttributes,
  LLMParameterAttributes,
  PromptTemplateAttributes,
  RetrievalDocument,
  SafeFunction,
  TokenCountAttributes,
  ToolAttributes,
} from "./types";

import { Attributes } from "@opentelemetry/api";

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

export const RETRIEVAL_DOCUMENTS =
  `${SemanticAttributePrefixes.retrieval}.${RetrievalAttributePostfixes.documents}` as const;

export class LlamaIndexInstrumentation extends InstrumentationBase<typeof llamaindex> {

  constructor(config?: InstrumentationConfig) {
    super(
      "@arizeai/openinference-instrumentation-llama-index",
      VERSION,
      Object.assign({}, config),
    );
  }

  public manuallyInstrument(module: typeof llamaindex) {
    diag.debug(`Manually instrumenting ${MODULE_NAME}`);
    this.patch(module);
  }

  protected init(): InstrumentationModuleDefinition<typeof llamaindex> {
    const module = new InstrumentationNodeModuleDefinition<typeof llamaindex>(
      "llamaindex",
      [">=0.1.0"],
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
    return module;
  }

  private patch(moduleExports: typeof llamaindex, moduleVersion?: string) {
    this._diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);
    if (_isOpenInferencePatched) {
      return moduleExports;
    }

    const instrumentation: LlamaIndexInstrumentation = this;

    type RetrieverQueryEngineQueryType =
      typeof moduleExports.RetrieverQueryEngine.prototype.query;

    this._wrap(
      moduleExports.RetrieverQueryEngine.prototype,
      "query",
      (original: RetrieverQueryEngineQueryType): any => {
        return this.patchQueryMethod(
          original,
          moduleExports,
          instrumentation
        );
      },
    );

    this._wrap(
      moduleExports.VectorIndexRetriever.prototype,
      "retrieve",
      (original): any => {
        return this.patchRetrieveMethod(
          original,
          moduleExports,
          instrumentation
        );
      },
    )

    _isOpenInferencePatched = true;

    return moduleExports;
  }

  private unpatch(moduleExports: typeof llamaindex, moduleVersion?: string) {
    this._diag.debug(`Un-patching ${MODULE_NAME}@${moduleVersion}`);
    this._unwrap(moduleExports.RetrieverQueryEngine.prototype, "query");

    _isOpenInferencePatched = false;
  }

  private patchQueryMethod(
    original: typeof module.RetrieverQueryEngine.prototype.query,
    module: typeof llamaindex,
    instrumentation: LlamaIndexInstrumentation
  ) {
    return function patchedQuery(
      this: unknown,
      ...args: Parameters<typeof module.RetrieverQueryEngine.prototype.query>
    ) {
      const span = instrumentation.tracer.startSpan(`query`, {
        kind: SpanKind.INTERNAL,
        attributes: {
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.CHAIN,
          [SemanticConventions.INPUT_VALUE]: args[0].query,
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
        });
        span.end();
        return result;
      });
      return context.bind(execContext, wrappedPromise);
    };
  }

  // RETRIEVAL
  // parse the retrieval document
    // content
    // ID
    // score
    // metadata

  // iterate through documents array
    // parse
    // set correct document attributes
  private patchRetrieveMethod(
    original: typeof module.VectorIndexRetriever.prototype.retrieve,
    module: typeof llamaindex,
    instrumentation: LlamaIndexInstrumentation
  ) {
    return function patchedRetrieve(
      this: unknown,
      ...args: Parameters<typeof module.VectorIndexRetriever.prototype.retrieve>
    ) {

      const span = instrumentation.tracer.startSpan(`retrieve`, {
        kind: SpanKind.INTERNAL,
        attributes: {
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.RETRIEVER,
          [SemanticConventions.INPUT_VALUE]: args[0].query,
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
        console.log(result);

        const docs: Attributes = {};
        result.forEach((document, index) => {
          const { node, score } = document;

          if (node instanceof TextNode) {
            const textNode = node as TextNode;
            const nodeId = textNode.id_;
            const nodeText = textNode.getContent();
            const nodeMetadata = textNode.metadata;

            docs[`${RETRIEVAL_DOCUMENTS}.${index}.document.id`] = nodeId;
            docs[`${RETRIEVAL_DOCUMENTS}.${index}.document.score`] = score;
            docs[`${RETRIEVAL_DOCUMENTS}.${index}.document.content`] = nodeText;
            docs[`${RETRIEVAL_DOCUMENTS}.${index}.document.metadata`] = safelyJSONStringify(nodeMetadata) ?? undefined;
          }
        });
        span.setAttributes(docs);

        span.end();
        return result;
      });
      return context.bind(execContext, wrappedPromise);
    };
  }
}
