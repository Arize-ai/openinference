import type { Span } from "@opentelemetry/api";
import { context, SpanStatusCode, trace } from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";

/**
 * Resolves the execution context for the current span
 * If tracing is suppressed, the span is dropped and the current context is returned
 * @param span
 */
export function getExecContext(span: Span) {
  const activeContext = context.active();
  const suppressTracing = isTracingSuppressed(activeContext);
  const execContext = suppressTracing ? trace.setSpan(context.active(), span) : activeContext;
  // Drop the span from the context
  if (suppressTracing) {
    trace.deleteSpan(activeContext);
  }
  return execContext;
}

/**
 * Records an error on the span and sets its status, without ending it.
 */
export function recordSpanError(span: Span, error: unknown) {
  if (error instanceof Error) {
    span.recordException(error);
    span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
  } else {
    span.recordException(String(error));
    span.setStatus({ code: SpanStatusCode.ERROR, message: String(error) });
  }
}
