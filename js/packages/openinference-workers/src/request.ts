import type { ExecutionContext } from "@cloudflare/workers-types";
import {
  context,
  diag,
  trace,
  INVALID_SPAN_CONTEXT,
  SpanStatusCode,
  type Attributes,
  type Span,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";

import type { OITracer } from "@arizeai/openinference-core";
import {
  SemanticConventions,
  type OpenInferenceSpanKind,
} from "@arizeai/openinference-semantic-conventions";

import { extractTraceContext } from "./propagation.js";

/**
 * One inbound request as one span: the parent comes from the request's `traceparent`, the
 * span is active for everything `fn` awaits, its status follows the outcome, and the export
 * is handed to `waitUntil` so the response is not held for the collector. Works for a Worker's
 * `fetch(request, env, ctx)` and for a Durable Object's `fetch(request)` with `this.ctx`.
 */
export interface RequestSpanOptions {
  tracer: OITracer;
  request: Pick<Request, "headers">;
  name: string;
  kind: OpenInferenceSpanKind | `${OpenInferenceSpanKind}`;
  attributes?: Attributes;
  /** `ctx` of a Worker handler or a Durable Object; without it the flush is awaited inline. */
  execution?: Pick<ExecutionContext, "waitUntil"> | null;
  flush: () => Promise<void>;
}

export async function withRequestSpan<T>(
  options: RequestSpanOptions,
  fn: (span: Span) => Promise<T>,
): Promise<T> {
  if (isTracingSuppressed(context.active())) {
    return fn(trace.wrapSpanContext(INVALID_SPAN_CONTEXT));
  }
  const parent = extractTraceContext({ headers: options.request.headers });
  const attributes: Attributes = {
    [SemanticConventions.OPENINFERENCE_SPAN_KIND]: options.kind,
    ...options.attributes,
  };
  const run = () =>
    options.tracer.startActiveSpan(options.name, { attributes }, async (span) => {
      try {
        const result = await fn(span);
        return result;
      } catch (e) {
        span.recordException(e instanceof Error ? e : new Error(String(e)));
        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: e instanceof Error ? e.message : String(e),
        });
        throw e;
      } finally {
        span.end();
      }
    });
  try {
    return await context.with(parent, run);
  } finally {
    const flushing = Promise.resolve()
      .then(() => options.flush())
      .catch((error: unknown) => {
        diag.warn("OpenInference request flush failed", error);
      });
    if (options.execution) options.execution.waitUntil(flushing);
    else await flushing;
  }
}
