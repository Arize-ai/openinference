import { SpanStatusCode } from "@opentelemetry/api";
import type { ReadableSpan, Span } from "@opentelemetry/sdk-trace-base";

import { getParentSpanId, isLikelyAISDKSpan } from "./typeUtils.js";
import { addOpenInferenceAttributesToSpan } from "./utils.js";

type TraceAggregate = {
  activeSpans: number;
  hadError: boolean;
  firstErrorMessage?: string;
  isAISDKTrace: boolean;
};

const spanHasErrorSignal = (span: ReadableSpan): { error: boolean; message?: string } => {
  if (span.status.code === SpanStatusCode.ERROR) {
    return { error: true, message: span.status.message };
  }

  const attributes = span.attributes as Record<string, unknown>;
  const finishReason = attributes["ai.response.finishReason"];
  if (finishReason === "error") {
    return { error: true, message: "ai.response.finishReason=error" };
  }

  const genAIFinishReasons = attributes["gen_ai.response.finish_reasons"];
  if (Array.isArray(genAIFinishReasons) && genAIFinishReasons.includes("error")) {
    return {
      error: true,
      message: "gen_ai.response.finish_reasons includes error",
    };
  }

  const hasExceptionEvent = span.events?.some((event) => event.name === "exception");
  if (hasExceptionEvent) {
    return { error: true, message: "exception" };
  }

  return { error: false };
};

const maybeSetRootStatus = ({
  span,
  traceAggregate,
}: {
  span: ReadableSpan;
  traceAggregate: TraceAggregate;
}): void => {
  // Called only for root spans; only set status when it's currently UNSET.
  if (!isLikelyAISDKSpan(span)) return;
  if (span.status.code !== SpanStatusCode.UNSET) return;

  // ReadableSpan is typed as readonly; runtime Span objects are mutable.
  Reflect.set(
    span,
    "status",
    traceAggregate.hadError
      ? { code: SpanStatusCode.ERROR, message: traceAggregate.firstErrorMessage }
      : { code: SpanStatusCode.OK },
  );
};

const maybeSetSpanOkStatus = (span: ReadableSpan): void => {
  // Set OK status on AI SDK spans that completed without error
  if (!isLikelyAISDKSpan(span)) return;
  if (span.status.code !== SpanStatusCode.UNSET) return;

  // ReadableSpan is typed as readonly; runtime Span objects are mutable.
  Reflect.set(span, "status", { code: SpanStatusCode.OK });
};

const maybeRenameRootSpan = ({
  span,
  isRootSpan,
}: {
  span: ReadableSpan;
  isRootSpan: boolean;
}): void => {
  if (!isRootSpan) return;
  if (!isLikelyAISDKSpan(span)) return;

  const attributes = span.attributes as Record<string, unknown>;
  const operationName = attributes["operation.name"];
  if (typeof operationName !== "string" || operationName.length === 0) return;
  if (span.name === operationName) return;
  // Preserve a framework wrapper's own ai.* span name (e.g. "ai.eve.turn") when its
  // operation.name is something unrelated (e.g. "eve") — renaming would clobber the meaningful
  // name with a worse one. This is narrow: it only skips when the span name is itself ai.* and
  // the operation.name is not. Native AI SDK spans (operation.name "ai.generateText <fnId>")
  // and gen_ai spans keep their existing rename behavior.
  if (span.name.startsWith("ai.") && !operationName.startsWith("ai.")) return;

  // NOTE: Span.updateName() refuses to update after end(); by the time span processors
  // run, spans are already ended. Assign directly.
  Reflect.set(span, "name", operationName);
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
    const traceAggregate = this.traceAggregates.get(traceId);
    if (traceAggregate) {
      traceAggregate.activeSpans += 1;
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

    const isRootSpan = getParentSpanId(span) == null;
    const traceId = span.spanContext().traceId;
    const traceAggregate = this.traceAggregates.get(traceId);

    // If we don't have an aggregate for this trace, just process the span
    if (traceAggregate == null) {
      maybeRenameRootSpan({ span, isRootSpan });
      return;
    }

    // Mark trace as AI SDK trace if this span is an AI SDK span
    // (attributes are available at onEnd time, not onStart)
    if (isLikelyAISDKSpan(span)) {
      traceAggregate.isAISDKTrace = true;
    }

    // Only aggregate error state for AI SDK traces
    if (traceAggregate.isAISDKTrace) {
      const { error, message } = spanHasErrorSignal(span);
      if (error) {
        traceAggregate.hadError = true;
        if (traceAggregate.firstErrorMessage == null && message != null) {
          traceAggregate.firstErrorMessage = message;
        }
      }
    }

    maybeRenameRootSpan({ span, isRootSpan });

    // Set status for AI SDK spans:
    // - Root spans get OK/ERROR based on aggregate error state
    // - Child spans get OK if they completed without error (already have ERROR if they errored)
    if (traceAggregate.isAISDKTrace) {
      if (isRootSpan) {
        maybeSetRootStatus({ span, traceAggregate });
      } else {
        maybeSetSpanOkStatus(span);
      }
    }

    // Decrement active span count and cleanup when the trace completes
    traceAggregate.activeSpans = Math.max(0, traceAggregate.activeSpans - 1);
    if (traceAggregate.activeSpans === 0) {
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
