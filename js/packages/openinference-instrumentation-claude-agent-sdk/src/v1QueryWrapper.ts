import type {
  Options as SDKOptions,
  SDKMessage,
  SDKUserMessage,
} from "@anthropic-ai/claude-agent-sdk";
import type { Span } from "@opentelemetry/api";
import { context, SpanStatusCode, trace } from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";

import type { OITracer } from "@arizeai/openinference-core";
import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { ToolSpanTracker, mergeHooks } from "./hookInjector";
import {
  extractInitAttributes,
  extractResultErrorAttributes,
  extractResultSuccessAttributes,
  formatPromptAttributes,
  isResultErrorMessage,
  isResultSuccessMessage,
  isSystemInitMessage,
} from "./messageProcessor";

/**
 * Parameters for the SDK query function.
 */
type QueryParams = {
  prompt: string | AsyncIterable<SDKUserMessage>;
  options?: SDKOptions;
};

/**
 * The SDK query function type — returns an async iterable of SDK messages.
 */
type QueryFunction = (params: QueryParams) => AsyncIterable<SDKMessage>;

/**
 * Creates a wrapped version of the SDK's `query()` function that produces
 * AGENT spans and TOOL child spans via hook injection.
 *
 * @param options.original - The original SDK `query()` function
 * @param options.oiTracer - OITracer instance for creating spans
 * @returns A wrapped query function with identical signature
 */
export function wrapQuery({
  original,
  oiTracer,
}: {
  original: QueryFunction;
  oiTracer: OITracer;
}): QueryFunction {
  return function wrappedQuery(params: QueryParams): AsyncIterable<SDKMessage> {
    const activeContext = context.active();
    if (isTracingSuppressed(activeContext)) {
      return original(params);
    }

    const inputAttrs = formatPromptAttributes(params.prompt);

    const toolTracker = new ToolSpanTracker(oiTracer);

    // We need to return an AsyncIterable that, when iterated, starts the span
    // and wraps each yielded message.
    return {
      [Symbol.asyncIterator]() {
        // Start the AGENT span when iteration begins
        const span: Span = oiTracer.startSpan(`ClaudeAgent.query`, {
          attributes: {
            [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
            ...inputAttrs,
          },
        });

        // Track whether an error result was received so we don't
        // overwrite ERROR status with OK on normal completion.
        let hasError = false;

        // Inject hooks into options
        const modifiedOptions = mergeHooks({
          options: params.options,
          toolTracker,
          parentSpan: span,
        });

        const innerIterable = original({
          ...params,
          options: modifiedOptions,
        });
        const innerIterator = innerIterable[Symbol.asyncIterator]();

        return {
          async next() {
            try {
              const result = await context.with(trace.setSpan(activeContext, span), () =>
                innerIterator.next(),
              );

              if (!result.done) {
                if (isResultErrorMessage(result.value)) {
                  hasError = true;
                }
                processMessage(result.value, span);
              }

              if (result.done) {
                // Generator completed normally
                toolTracker.endAllInFlight();
                if (!hasError) {
                  span.setStatus({ code: SpanStatusCode.OK });
                }
                span.end();
              }

              return result;
            } catch (error) {
              toolTracker.endAllInFlight();
              if (error instanceof Error) {
                span.recordException(error);
                span.setStatus({
                  code: SpanStatusCode.ERROR,
                  message: error.message,
                });
              }
              span.end();
              throw error;
            }
          },
          async return(value?: unknown) {
            // Generator abandoned early (e.g., break)
            toolTracker.endAllInFlight();
            if (!hasError) {
              span.setStatus({ code: SpanStatusCode.OK });
            }
            span.end();
            if (innerIterator.return) {
              return innerIterator.return(value);
            }
            return { done: true as const, value: undefined };
          },
          async throw(error?: unknown) {
            toolTracker.endAllInFlight();
            if (error instanceof Error) {
              span.recordException(error);
              span.setStatus({
                code: SpanStatusCode.ERROR,
                message: error.message,
              });
            }
            span.end();
            if (innerIterator.throw) {
              return innerIterator.throw(error);
            }
            throw error;
          },
        };
      },
    };
  };
}

/**
 * Processes a message from the SDK generator, setting span attributes
 * based on message type.
 */
function processMessage(msg: SDKMessage, span: Span): void {
  if (isSystemInitMessage(msg)) {
    const { sessionId, model } = extractInitAttributes(msg);
    span.setAttributes({
      [SemanticConventions.SESSION_ID]: sessionId,
      [SemanticConventions.LLM_MODEL_NAME]: model,
    });
  } else if (isResultSuccessMessage(msg)) {
    span.setAttributes(extractResultSuccessAttributes(msg));
  } else if (isResultErrorMessage(msg)) {
    span.setAttributes(extractResultErrorAttributes(msg));
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: `Result error: ${msg.subtype}`,
    });
  }
}
