import { SpanStatusCode } from "@opentelemetry/api";
import { ReadableSpan, Span } from "@opentelemetry/sdk-trace-base";

import { addOpenInferenceAttributesToSpan } from "./utils";

type TraceAggregate = {
  activeSpans: number;
  hadError: boolean;
  firstErrorMessage?: string;
  isAISDKTrace: boolean;
};

const isLikelyAISDKSpan = (span: ReadableSpan | Span): boolean => {
  const attrs = span.attributes as Record<string, unknown> | undefined;
  const opName = attrs?.["operation.name"];
  const opId = attrs?.["ai.operationId"];

  if (typeof opName === "string" && opName.startsWith("ai.")) return true;
  if (typeof opId === "string" && opId.startsWith("ai.")) return true;

  // gen_ai.* indicates AI SDK v6+ GenAI spans
  return (
    attrs != null && Object.keys(attrs).some((k) => k.startsWith("gen_ai."))
  );
};

const spanHasErrorSignal = (
  span: ReadableSpan,
): { error: boolean; message?: string } => {
  if (span.status.code === SpanStatusCode.ERROR) {
    return { error: true, message: span.status.message };
  }

  const attrs = span.attributes as Record<string, unknown>;
  const finishReason = attrs["ai.response.finishReason"];
  if (finishReason === "error") {
    return { error: true, message: "ai.response.finishReason=error" };
  }

  const genFinishReasons = attrs["gen_ai.response.finish_reasons"];
  if (Array.isArray(genFinishReasons) && genFinishReasons.includes("error")) {
    return {
      error: true,
      message: "gen_ai.response.finish_reasons includes error",
    };
  }

  const hasExceptionEvent = span.events?.some((e) => e.name === "exception");
  if (hasExceptionEvent) {
    return { error: true, message: "exception" };
  }

  return { error: false };
};

const maybeSetRootStatus = (span: ReadableSpan, agg: TraceAggregate): void => {
  // Only set status on the root span, and only when it's currently UNSET.
  if (span.parentSpanId != null) return;
  if (!isLikelyAISDKSpan(span)) return;
  if (span.status.code !== SpanStatusCode.UNSET) return;

  // ReadableSpan is typed as readonly; runtime Span objects are mutable.
  (
    span as unknown as { status: { code: SpanStatusCode; message?: string } }
  ).status = agg.hadError
    ? { code: SpanStatusCode.ERROR, message: agg.firstErrorMessage }
    : { code: SpanStatusCode.OK };
};

const maybeSetSpanOkStatus = (span: ReadableSpan): void => {
  // Set OK status on AI SDK spans that completed without error
  if (!isLikelyAISDKSpan(span)) return;
  if (span.status.code !== SpanStatusCode.UNSET) return;

  // ReadableSpan is typed as readonly; runtime Span objects are mutable.
  (
    span as unknown as { status: { code: SpanStatusCode; message?: string } }
  ).status = { code: SpanStatusCode.OK };
};

const maybeRenameRootSpan = (span: ReadableSpan): void => {
  if (span.parentSpanId != null) return;
  if (!isLikelyAISDKSpan(span)) return;

  const attrs = span.attributes as Record<string, unknown>;
  const operationName = attrs["operation.name"];
  if (typeof operationName !== "string" || operationName.length === 0) return;
  if (span.name === operationName) return;

  // NOTE: Span.updateName() refuses to update after end(); by the time span processors
  // run, spans are already ended. Assign directly.
  (span as unknown as { name: string }).name = operationName;
};

/**
 * Manages trace-level aggregate state for error propagation and span tracking.
 *
 * Tracking strategy to prevent memory leaks:
 * - All spans are tracked at onStart (we need the count for cleanup)
 * - At onEnd, we check if the span is an AI SDK span and mark the trace accordingly
 * - Error state is only aggregated for AI SDK traces
 * - Non-AI SDK traces are cleaned up immediately when they complete
 */
export class TraceAggregateManager {
  private readonly traceAggregates = new Map<string, TraceAggregate>();

  /**
   * Track a span starting. All spans are tracked to maintain accurate counts.
   */
  onStart(span: Span): void {
    const traceId = span.spanContext().traceId;
    const agg = this.traceAggregates.get(traceId);
    if (agg) {
      agg.activeSpans += 1;
    } else {
      this.traceAggregates.set(traceId, {
        activeSpans: 1,
        hadError: false,
        isAISDKTrace: false,
      });
    }
  }

  /**
   * Process a span ending: update error state, rename root span, set root status.
   */
  onEnd(span: ReadableSpan): void {
    addOpenInferenceAttributesToSpan(span);

    const traceId = span.spanContext().traceId;
    const agg = this.traceAggregates.get(traceId);

    // If we don't have an aggregate for this trace, just process the span
    if (agg == null) {
      maybeRenameRootSpan(span);
      return;
    }

    // Mark trace as AI SDK trace if this span is an AI SDK span
    // (attributes are available at onEnd time, not onStart)
    if (isLikelyAISDKSpan(span)) {
      agg.isAISDKTrace = true;
    }

    // Only aggregate error state for AI SDK traces
    if (agg.isAISDKTrace) {
      const { error, message } = spanHasErrorSignal(span);
      if (error) {
        agg.hadError = true;
        if (agg.firstErrorMessage == null && message != null) {
          agg.firstErrorMessage = message;
        }
      }
    }

    maybeRenameRootSpan(span);

    // Set status for AI SDK spans:
    // - Root spans get OK/ERROR based on aggregate error state
    // - Child spans get OK if they completed without error (already have ERROR if they errored)
    if (agg.isAISDKTrace) {
      if (span.parentSpanId == null) {
        maybeSetRootStatus(span, agg);
      } else {
        maybeSetSpanOkStatus(span);
      }
    }

    // Decrement active span count and cleanup when the trace completes
    agg.activeSpans = Math.max(0, agg.activeSpans - 1);
    if (agg.activeSpans === 0) {
      this.traceAggregates.delete(traceId);
    }
  }

  /**
   * Clear all tracked aggregates. Called during shutdown/forceFlush to prevent leaks.
   */
  clear(): void {
    this.traceAggregates.clear();
  }
}
