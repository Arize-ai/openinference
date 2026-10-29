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
  extractAssistantStopReason,
  extractInitAttributes,
  extractResultErrorAttributes,
  extractResultSuccessAttributes,
  formatPromptAttributes,
  isAssistantMessage,
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
 * The SDK query function type — returns an async iterable of SDK messages. At runtime the
 * SDK returns a `Query`, which is also an `AsyncGenerator` with control methods
 * (`interrupt`, `setModel`, …); the wrapper preserves that richer object.
 */
type QueryFunction = (params: QueryParams) => AsyncIterable<SDKMessage>;

/**
 * Creates a wrapped version of the SDK's `query()` function that produces
 * AGENT spans and TOOL child spans via hook injection.
 *
 * The wrapped function stays observationally equivalent to the unwrapped one: it calls
 * the SDK eagerly (the SDK spawns the Claude Code process at call time, not at first
 * iteration) and returns a proxy over the real `Query`, so control methods such as
 * `interrupt()` keep working. Only the async iteration is traced.
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

    // The AGENT span covers the whole query, so it starts when the SDK starts working.
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

    const endWithError = (error: unknown): void => {
      toolTracker.endAllInFlight();
      if (error instanceof Error) {
        span.recordException(error);
        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: error.message,
        });
      }
      span.end();
    };

    let innerQuery: AsyncIterable<SDKMessage>;
    try {
      innerQuery = original({ ...params, options: modifiedOptions });
    } catch (error) {
      // A synchronous throw has to keep the span balanced before it propagates.
      endWithError(error);
      throw error;
    }

    const innerIterator = innerQuery[Symbol.asyncIterator]();

    const iterator: AsyncIterator<SDKMessage> & AsyncIterable<SDKMessage> = {
      async next(...args: [] | [unknown]) {
        try {
          const result = await context.with(trace.setSpan(activeContext, span), () =>
            innerIterator.next(...args),
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
          endWithError(error);
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
        endWithError(error);
        if (innerIterator.throw) {
          return innerIterator.throw(error);
        }
        throw error;
      },
      [Symbol.asyncIterator]() {
        return iterator;
      },
    };

    // Delegate everything the SDK puts on `Query` (interrupt, setModel,
    // setPermissionMode, applyFlagSettings, reinitialize, initializationResult, …) to the
    // real object so enabling tracing never removes control methods. Only iteration is
    // replaced with the traced version above.
    return new Proxy(innerQuery, {
      get(target, property, receiver) {
        if (property === Symbol.asyncIterator) {
          return () => iterator;
        }
        if (property === "next") {
          return iterator.next.bind(iterator);
        }
        if (property === "return" && iterator.return) {
          return iterator.return.bind(iterator);
        }
        if (property === "throw" && iterator.throw) {
          return iterator.throw.bind(iterator);
        }
        const value = Reflect.get(target, property, receiver);
        // SDK methods rely on `this`, so bind them to the real query object.
        return typeof value === "function" ? value.bind(target) : value;
      },
    });
  };
}

/**
 * Processes a message from the SDK generator, setting span attributes
 * based on message type.
 */
function processMessage(msg: SDKMessage, span: Span): void {
  if (isAssistantMessage(msg)) {
    const stopReason = extractAssistantStopReason(msg);
    if (stopReason != null) {
      span.setAttribute(SemanticConventions.LLM_FINISH_REASON, stopReason);
    }
  } else if (isSystemInitMessage(msg)) {
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
