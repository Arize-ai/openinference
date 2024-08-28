import {
  BatchSpanProcessor,
  BufferConfig,
  ReadableSpan,
  SimpleSpanProcessor,
  SpanExporter,
} from "@opentelemetry/sdk-trace-base";
import { addOpenInferenceAttributesToSpan, shouldExportSpan } from "./utils";

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
  private readonly onlyExportOpenInferenceSpans: boolean;
  constructor({
    exporter,
    onlyExportOpenInferenceSpans = true,
  }: {
    /**
     * The exporter to pass spans to.
     */
    readonly exporter: SpanExporter;
    /**
     * Whether or not to only export OpenInference spans.
     * @default true
     */
    readonly onlyExportOpenInferenceSpans?: boolean;
  }) {
    super(exporter);
    this.onlyExportOpenInferenceSpans = onlyExportOpenInferenceSpans;
  }

  onEnd(span: ReadableSpan): void {
    addOpenInferenceAttributesToSpan(span);
    if (
      shouldExportSpan({
        span,
        onlyExportOpenInferenceSpans: this.onlyExportOpenInferenceSpans,
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
  private readonly onlyExportOpenInferenceSpans: boolean;
  constructor({
    exporter,
    onlyExportOpenInferenceSpans = true,
    config,
  }: {
    /**
     * The exporter to pass spans to.
     */
    readonly exporter: SpanExporter;
    /**
     * Whether or not to only export OpenInference spans.
     * @default true
     */
    readonly onlyExportOpenInferenceSpans?: boolean;
    /**
     * The configuration options for processor.
     */
    config?: BufferConfig;
  }) {
    super(exporter, config);
    this.onlyExportOpenInferenceSpans = onlyExportOpenInferenceSpans;
  }

  onEnd(span: ReadableSpan): void {
    addOpenInferenceAttributesToSpan(span);
    if (
      shouldExportSpan({
        span,
        onlyExportOpenInferenceSpans: this.onlyExportOpenInferenceSpans,
      })
    ) {
      super.onEnd(span);
    }
  }
}
