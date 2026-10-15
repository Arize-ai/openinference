import type { APIPromise } from "@anthropic-ai/sdk";
import type { Span } from "@opentelemetry/api";
import { context, SpanStatusCode, trace } from "@opentelemetry/api";
import { safeExecuteInTheMiddle } from "@opentelemetry/instrumentation";

import { getExecContext, recordSpanError } from "./spanLifecycle";

/**
 * Ends a span at most once, and records an error on it before it ends.
 */
interface SpanLifecycle {
  endSpan: () => void;
  recordError: (error: unknown) => void;
}

/**
 * Calls an SDK method returning an `APIPromise` in the span's context, and
 * hands its parsed result to `onResult`, which records it and ends the span.
 *
 * The result stays an `APIPromise` (through `_thenUnwrap`) so `withResponse()`
 * and `asResponse()` keep working, which `client.messages.stream()` relies on.
 * A request failure is recorded without triggering the parse, and a caller
 * that takes the raw response ends the span, as nothing is parsed for it.
 */
export function traceAnthropicAPIPromise<T>({
  span,
  invoke,
  onResult,
}: {
  span: Span;
  invoke: () => PromiseLike<T>;
  onResult: (result: T, lifecycle: SpanLifecycle) => T;
}) {
  const execContext = getExecContext(span);
  // The span can be ended by the parse path, the asResponse() override, or
  // an error, so guard against ending it more than once.
  let spanEnded = false;
  const endSpan = () => {
    if (spanEnded) {
      return;
    }
    spanEnded = true;
    span.end();
  };
  const recordError = (error: unknown) => {
    if (spanEnded) {
      return;
    }
    recordSpanError(span, error);
    endSpan();
  };
  const lifecycle: SpanLifecycle = { endSpan, recordError };

  const execPromise = safeExecuteInTheMiddle(
    () => context.with(trace.setSpan(execContext, span), invoke),
    (error: Error | undefined) => {
      // Push the error to the span
      if (error) {
        recordError(error);
      }
    },
  );
  const handleResult = (result: T) => onResult(result, lifecycle);

  if (hasThenUnwrap(execPromise)) {
    const wrappedPromise = execPromise._thenUnwrap(handleResult);
    const rawResponse = wrappedPromise.asResponse.bind(wrappedPromise);

    // Record request failures without triggering parse (which would consume
    // the body). Covers the await/withResponse and asResponse paths.
    rawResponse().catch(recordError);

    // Wrap asResponse() itself so the span is finalized only when the caller
    // actually chooses the raw-response path. Those callers bypass parsing,
    // so no parsed output attributes are available.
    wrappedPromise.asResponse = async () => {
      const response = await rawResponse();
      span.setStatus({ code: SpanStatusCode.OK });
      endSpan();
      return response;
    };

    // withResponse() calls this.asResponse() internally; reimplement it
    // against the raw response so the override above doesn't end the span
    // before the result is recorded.
    wrappedPromise.withResponse = async () => {
      const [data, response] = await Promise.all([
        wrappedPromise.then((value) => value),
        rawResponse(),
      ]);
      return {
        data,
        response,
        request_id: response.headers.get("request-id"),
        workspace_id: response.headers.get("anthropic-workspace-id"),
      };
    };

    return context.bind(execContext, wrappedPromise);
  }

  const wrappedPromise = Promise.resolve(execPromise)
    .then(handleResult)
    .catch((error: unknown) => {
      recordError(error);
      throw error;
    });
  return context.bind(execContext, wrappedPromise);
}

/**
 * True when create() returned an APIPromise we can transform with _thenUnwrap.
 */
function hasThenUnwrap<T>(promise: PromiseLike<T>): promise is APIPromise<T> {
  return "_thenUnwrap" in promise && typeof promise._thenUnwrap === "function";
}
