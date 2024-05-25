/* eslint-disable @typescript-eslint/no-explicit-any */
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

const MODULE_NAME = "llamaindex";

/**
 * Flag to check if the openai module has been patched
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
} from "@arizeai/openinference-semantic-conventions";

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
export class LlamaIndexInstrumentation extends InstrumentationBase<
  typeof llamaindex
> {
  constructor(config?: InstrumentationConfig) {
    super(
      "@arizeai/openinference-instrumentation-llama-index",
      VERSION,
      Object.assign({}, config),
    );
  }
  manuallyInstrument(module: typeof llamaindex) {
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

    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const instrumentation: LlamaIndexInstrumentation = this;

    type RetrieverQueryEngineQueryType =
      typeof moduleExports.RetrieverQueryEngine.prototype.query;

    this._wrap(
      moduleExports.RetrieverQueryEngine.prototype,
      "query",
      (original: RetrieverQueryEngineQueryType): any => {
        return function patchedQuery(
          this: unknown,
          ...args: Parameters<RetrieverQueryEngineQueryType>
        ) {
          const span = instrumentation.tracer.startSpan(`Query`, {
            kind: SpanKind.INTERNAL,
            attributes: {
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                OpenInferenceSpanKind.CHAIN,
              [SemanticConventions.INPUT_VALUE]: args[0].query,
            },
          });

          const execContext = getExecContext(span);

          const execPromise = safeExecuteInTheMiddle<
            ReturnType<RetrieverQueryEngineQueryType>
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
            // if (isChatCompletionResponse(result)) {
            //   // Record the results
            //   span.setAttributes({
            //     [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(result),
            //     [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
            //     // Override the model from the value sent by the server
            //     [SemanticConventions.LLM_MODEL_NAME]: result.model,
            //     ...getChatCompletionLLMOutputMessagesAttributes(result),
            //     ...getUsageAttributes(result),
            //   });
            //   span.setStatus({ code: SpanStatusCode.OK });
            //   span.end();
            // } else {
            //   // This is a streaming response
            //   // handle the chunks and add them to the span
            //   // First split the stream via tee
            //   const [leftStream, rightStream] = result.tee();
            //   consumeChatCompletionStreamChunks(rightStream, span);
            //   result = leftStream;
            // }

            span.end();
            return result;
          });
          return context.bind(execContext, wrappedPromise);
        };
      },
    );

    _isOpenInferencePatched = true;

    return moduleExports;
  }

  private unpatch(moduleExports: typeof llamaindex, moduleVersion?: string) {
    this._diag.debug(`Un-patching ${MODULE_NAME}@${moduleVersion}`);
    this._unwrap(moduleExports.RetrieverQueryEngine.prototype, "query");

    _isOpenInferencePatched = false;
  }
}
