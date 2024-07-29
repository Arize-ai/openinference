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
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";

import {
  TextNode
} from "llamaindex";

import {
  OpenInferenceSpanKind,
  SemanticConventions,
  RetrievalAttributePostfixes,
  SemanticAttributePrefixes,
} from "@arizeai/openinference-semantic-conventions";

import { Attributes } from "@opentelemetry/api";

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

    this._wrap(
      moduleExports.RetrieverQueryEngine.prototype,
      "query",
      (original): any => {
        return this.patchQueryMethod(
          original,
          moduleExports,
          this
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
          this
        );
      },
    )

    _isOpenInferencePatched = true;
    return moduleExports;
  }

  private unpatch(moduleExports: typeof llamaindex, moduleVersion?: string) {
    this._diag.debug(`Un-patching ${MODULE_NAME}@${moduleVersion}`);
    this._unwrap(moduleExports.RetrieverQueryEngine.prototype, "query");
    this._unwrap(moduleExports.VectorIndexRetriever.prototype, "retrieve");

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

  private patchRetrieveMethod(
    original: typeof module.VectorIndexRetriever.prototype.retrieve,
    module: typeof llamaindex,
    instrumentation: LlamaIndexInstrumentation
  ) {
    return function patchedRetrieve(
      this: unknown,
      ...args: Parameters<typeof module.VectorIndexRetriever.prototype.retrieve>
    ) {

      // Start the span with initial attributes: kind, span_kind, input
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
            docs[`${RETRIEVAL_DOCUMENTS}.${index}.document.metadata`] = JSON.stringify(nodeMetadata) ?? undefined;
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
