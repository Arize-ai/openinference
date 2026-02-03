import { Context, SpanStatusCode } from "@opentelemetry/api";
import {
  BatchSpanProcessor,
  BufferConfig,
  ReadableSpan,
  SimpleSpanProcessor,
  Span,
  SpanExporter,
} from "@opentelemetry/sdk-trace-base";

import { SpanFilter } from "./types";
import { addOpenInferenceAttributesToSpan, shouldExportSpan } from "./utils";

type TraceAggregate = {
  activeSpans: number;
  hadError: boolean;
  firstErrorMessage?: string;
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

/**
 * Extends {@link SimpleSpanProcessor} to support OpenInference attributes.
 * This processor enhances spans with OpenInference attributes before exporting them.
 * It can be configured to selectively export only OpenInference spans or all spans.
 * @extends {SimpleSpanProcessor}
 *
 * @example
 * ```typescript
 * import { OpenInferenceSimpleSpanProcessor, isOpenInferenceSpan } from "@arizeai/openinference-vercel";
 * import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto"
 *
 * const exporter = new OTLPTraceExporter();
 * const processor = new OpenInferenceSimpleSpanProcessor({
 *   exporter,
 *   spanFilter: isOpenInferenceSpan,
 * });
 *
 * const tracerProvider = new NodeTracerProvider({
 *   resource: resourceFromAttributes({
 *     [SEMRESATTRS_PROJECT_NAME]: "your-project-name",
 *   }),
 *   spanProcessors: [processor], // <-- pass processor here
 * });
 *
 * ```
 */
export class OpenInferenceSimpleSpanProcessor extends SimpleSpanProcessor {
  private readonly spanFilter?: SpanFilter;
  private readonly traceAggregates = new Map<string, TraceAggregate>();
  constructor({
    exporter,
    spanFilter,
  }: {
    /**
     * The exporter to pass spans to.
     */
    readonly exporter: SpanExporter;
    /**
     * A filter to apply to spans before exporting. If it returns true for a given span, that span will be exported.
     */
    readonly spanFilter?: SpanFilter;

    config?: BufferConfig;
  }) {
    super(exporter);
    this.spanFilter = spanFilter;
  }

  onStart(span: Span, parentContext: Context): void {
    const traceId = span.spanContext().traceId;
    const agg = this.traceAggregates.get(traceId);
    if (agg) {
      agg.activeSpans += 1;
    } else {
      this.traceAggregates.set(traceId, { activeSpans: 1, hadError: false });
    }
    super.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    addOpenInferenceAttributesToSpan(span);

    const traceId = span.spanContext().traceId;
    const agg =
      this.traceAggregates.get(traceId) ??
      ({ activeSpans: 0, hadError: false } satisfies TraceAggregate);

    const { error, message } = spanHasErrorSignal(span);
    if (error) {
      agg.hadError = true;
      if (agg.firstErrorMessage == null && message != null) {
        agg.firstErrorMessage = message;
      }
    }

    maybeSetRootStatus(span, agg);

    // Decrement active span count and cleanup when the trace completes.
    if (this.traceAggregates.has(traceId)) {
      agg.activeSpans = Math.max(0, agg.activeSpans - 1);
      if (agg.activeSpans === 0) {
        this.traceAggregates.delete(traceId);
      } else {
        this.traceAggregates.set(traceId, agg);
      }
    }

    if (
      shouldExportSpan({
        span,
        spanFilter: this.spanFilter,
      })
    ) {
      super.onEnd(span);
    }
  }
}

/**
 * Extends {@link BatchSpanProcessor} to support OpenInference attributes.
 * This processor enhances spans with OpenInference attributes before exporting them.
 * It can be configured to selectively export only OpenInference spans or all spans.
 * @extends {BatchSpanProcessor}
 *
 * @example
 * ```typescript
 * import { OpenInferenceBatchSpanProcessor, isOpenInferenceSpan } from "@arizeai/openinference-vercel";
 * import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto"
 *
 * const exporter = new OTLPTraceExporter();
 * const processor = new OpenInferenceBatchSpanProcessor({
 *   exporter,
 *   spanFilter: isOpenInferenceSpan,
 *   config: { maxQueueSize: 2048, scheduledDelayMillis: 5000 },
 * });
 *
 * const tracerProvider = new NodeTracerProvider({
 *   resource: resourceFromAttributes({
 *     [SEMRESATTRS_PROJECT_NAME]: "your-project-name",
 *   }),
 *   spanProcessors: [processor], // <-- pass processor here
 * });
 *
 * ```
 */
export class OpenInferenceBatchSpanProcessor extends BatchSpanProcessor {
  private readonly spanFilter?: SpanFilter;
  private readonly traceAggregates = new Map<string, TraceAggregate>();
  constructor({
    exporter,
    spanFilter,
    config,
  }: {
    /**
     * The exporter to pass spans to.
     */
    readonly exporter: SpanExporter;
    /**
     * A filter to apply to spans before exporting. If it returns true for a given span, that span will be exported.
     */
    readonly spanFilter?: SpanFilter;
    /**
     * The configuration options for processor.
     */
    config?: BufferConfig;
  }) {
    super(exporter, config);
    this.spanFilter = spanFilter;
  }

  forceFlush(): Promise<void> {
    return super.forceFlush();
  }

  shutdown(): Promise<void> {
    return super.shutdown();
  }

  onStart(_span: Span, _parentContext: Context): void {
    const traceId = _span.spanContext().traceId;
    const agg = this.traceAggregates.get(traceId);
    if (agg) {
      agg.activeSpans += 1;
    } else {
      this.traceAggregates.set(traceId, { activeSpans: 1, hadError: false });
    }
    return super.onStart(_span, _parentContext);
  }

  onEnd(span: ReadableSpan): void {
    addOpenInferenceAttributesToSpan(span);

    const traceId = span.spanContext().traceId;
    const agg =
      this.traceAggregates.get(traceId) ??
      ({ activeSpans: 0, hadError: false } satisfies TraceAggregate);

    const { error, message } = spanHasErrorSignal(span);
    if (error) {
      agg.hadError = true;
      if (agg.firstErrorMessage == null && message != null) {
        agg.firstErrorMessage = message;
      }
    }

    maybeSetRootStatus(span, agg);

    if (this.traceAggregates.has(traceId)) {
      agg.activeSpans = Math.max(0, agg.activeSpans - 1);
      if (agg.activeSpans === 0) {
        this.traceAggregates.delete(traceId);
      } else {
        this.traceAggregates.set(traceId, agg);
      }
    }

    if (
      shouldExportSpan({
        span,
        spanFilter: this.spanFilter,
      })
    ) {
      super.onEnd(span);
    }
  }
}
