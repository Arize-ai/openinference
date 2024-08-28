import {
  BatchSpanProcessor,
  BufferConfig,
  ReadableSpan,
  SimpleSpanProcessor,
  SpanExporter,
} from "@opentelemetry/sdk-trace-base";
import { addOpenInferenceAttributesToSpan, shouldExportSpan } from "./utils";
import { SpanFilter } from "./types";

/**
 * Extends {@link SimpleSpanProcessor} to support OpenInference attributes.
 * This processor enhances spans with OpenInference attributes before exporting them.
 * It can be configured to selectively export only OpenInference spans or all spans.
 * @extends {SimpleSpanProcessor}
 *
 * @example
 * ```typescript
 * const exporter = new OTLPTraceExporter();
 * const processor = new OpenInferenceSimpleSpanProcessor({
 *   exporter,
 *   onlyExportOpenInferenceSpans: true,
 * });
 * tracerProvider.addSpanProcessor(processor);
 * ```
 */
export class OpenInferenceSimpleSpanProcessor extends SimpleSpanProcessor {
  private readonly spanFilters?: SpanFilter[];
  constructor({
    exporter,
    spanFilters,
  }: {
    /**
     * The exporter to pass spans to.
     */
    readonly exporter: SpanExporter;
    /**
     * A list of filters to apply to spans before exporting. If at least one filter returns true for a given span, the span will be exported.
     */
    readonly spanFilters?: SpanFilter[];

    config?: BufferConfig;
  }) {
    super(exporter);
    this.spanFilters = spanFilters;
  }

  onEnd(span: ReadableSpan): void {
    addOpenInferenceAttributesToSpan(span);
    if (
      shouldExportSpan({
        span,
        spanFilters: this.spanFilters,
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
 * const exporter = new OTLPTraceExporter();
 * const processor = new OpenInferenceBatchSpanProcessor({
 *   exporter,
 *   onlyExportOpenInferenceSpans: true,
 *   config: { maxQueueSize: 2048, scheduledDelayMillis: 5000 }
 * });
 * tracerProvider.addSpanProcessor(processor);
 * ```
 */
export class OpenInferenceBatchSpanProcessor extends BatchSpanProcessor {
  private readonly spanFilters?: SpanFilter[];
  constructor({
    exporter,
    spanFilters,
    config,
  }: {
    /**
     * The exporter to pass spans to.
     */
    readonly exporter: SpanExporter;
    /**
     * A list of filters to apply to spans before exporting. If at least one filter returns true for a given span, the span will be exported.
     */
    readonly spanFilters?: SpanFilter[];
    /**
     * The configuration options for processor.
     */
    config?: BufferConfig;
  }) {
    super(exporter, config);
    this.spanFilters = spanFilters;
  }

  onEnd(span: ReadableSpan): void {
    addOpenInferenceAttributesToSpan(span);
    if (
      shouldExportSpan({
        span,
        spanFilters: this.spanFilters,
      })
    ) {
      super.onEnd(span);
    }
  }
}
