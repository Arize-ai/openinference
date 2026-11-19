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
 * The SDK query function type. The real SDK returns a `Query`, an
 * AsyncGenerator that also carries control methods such as `interrupt()`;
 * the wrapper hands back whatever the original returns.
 */
type QueryFunction<TQuery extends AsyncIterable<SDKMessage>> = (params: QueryParams) => TQuery;

/**
 * The traced view of the SDK's message iterator.
 */
type TracedIterator = {
  next(): Promise<IteratorResult<SDKMessage>>;
  return(value?: unknown): Promise<IteratorResult<SDKMessage>>;
  throw(error?: unknown): Promise<IteratorResult<SDKMessage>>;
  [Symbol.asyncIterator](): TracedIterator;
};

/**
 * Creates a wrapped version of the SDK's `query()` function that produces
 * AGENT spans and TOOL child spans via hook injection.
 *
 * The wrapper calls the SDK at `query()` time, as the unwrapped SDK does (it
 * spawns the Claude Code process on call), and returns the SDK's own `Query`
 * object behind a Proxy. Iteration goes through the traced iterator whether
 * the caller uses `for await`, `next()`/`return()`/`throw()` directly, or
 * `await using` (`Symbol.asyncDispose`), and every other member
 * (`interrupt()`, `setPermissionMode()`, ...) forwards to the SDK object
 * unchanged.
 *
 * @param options.original - The original SDK `query()` function
 * @param options.oiTracer - OITracer instance for creating spans
 * @returns A wrapped query function with identical signature
 */
export function wrapQuery<TQuery extends AsyncIterable<SDKMessage>>({
  original,
  oiTracer,
}: {
  original: QueryFunction<TQuery>;
  oiTracer: OITracer;
}): QueryFunction<TQuery> {
  return function wrappedQuery(params: QueryParams): TQuery {
    const activeContext = context.active();
    if (isTracingSuppressed(activeContext)) {
      return original(params);
    }

    const inputAttrs = formatPromptAttributes(params.prompt);
    const toolTracker = new ToolSpanTracker(oiTracer);

    // The AGENT span starts now: the hooks that produce TOOL spans need it as
    // their parent before the SDK is called, and the SDK is called now so that
    // its Query object exists to be returned.
    const span: Span = oiTracer.startSpan(`ClaudeAgent.query`, {
      attributes: {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
        ...inputAttrs,
      },
    });

    // Inject hooks into options
    const modifiedOptions = mergeHooks({
      options: params.options,
      toolTracker,
      parentSpan: span,
    });

    // Track whether an error result was received so we don't
    // overwrite ERROR status with OK on normal completion.
    let hasError = false;

    // The span can be ended from several paths (a synchronous SDK error,
    // completion, an iteration error, return(), throw(), dispose); only the
    // first one records status and ends it.
    let ended = false;
    const endSpan = (failure?: { error: unknown }): void => {
      if (ended) {
        return;
      }
      ended = true;
      toolTracker.endAllInFlight();
      if (failure) {
        if (failure.error instanceof Error) {
          span.recordException(failure.error);
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: failure.error.message,
          });
        }
      } else if (!hasError) {
        span.setStatus({ code: SpanStatusCode.OK });
      }
      span.end();
    };

    let query: TQuery;
    try {
      query = original({
        ...params,
        options: modifiedOptions,
      });
    } catch (error) {
      endSpan({ error });
      throw error;
    }

    // One traced iterator per query, created on first use and shared by every
    // entry point.
    let tracedIterator: TracedIterator | undefined;
    const getTracedIterator = (): TracedIterator => {
      if (tracedIterator) {
        return tracedIterator;
      }
      const innerIterator = query[Symbol.asyncIterator]();
      tracedIterator = {
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
              endSpan();
            }

            return result;
          } catch (error) {
            endSpan({ error });
            throw error;
          }
        },
        async return(value?: unknown) {
          // Generator abandoned early (e.g., break)
          endSpan();
          if (innerIterator.return) {
            return innerIterator.return(value);
          }
          return { done: true as const, value: undefined };
        },
        async throw(error?: unknown) {
          endSpan({ error });
          if (innerIterator.throw) {
            return innerIterator.throw(error);
          }
          throw error;
        },
        [Symbol.asyncIterator]() {
          return this;
        },
      };
      return tracedIterator;
    };

    return new Proxy(query, {
      get(target, prop) {
        if (prop === Symbol.asyncIterator) {
          return getTracedIterator;
        }
        if (prop === "next") {
          return () => getTracedIterator().next();
        }
        if (prop === "return") {
          return (value?: unknown) => getTracedIterator().return(value);
        }
        if (prop === "throw") {
          return (error?: unknown) => getTracedIterator().throw(error);
        }
        if (prop === Symbol.asyncDispose) {
          // `await using q = query(...)` disposes through the traced return()
          // so the span ends; the SDK's own dispose does the same return().
          return async () => {
            await getTracedIterator().return(undefined);
          };
        }
        // Everything else is the SDK's own member. Bind methods to the real
        // object so implementations that rely on `this` keep working.
        const value: unknown = Reflect.get(target, prop, target);
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
