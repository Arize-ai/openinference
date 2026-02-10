import { Context } from "@opentelemetry/api";
import {
  BatchSpanProcessor,
  BufferConfig,
  ReadableSpan,
  SimpleSpanProcessor,
  Span,
  SpanExporter,
} from "@opentelemetry/sdk-trace-base";

import { TraceAggregateManager } from "./TraceAggregateManager";
import { SpanFilter } from "./types";
import { shouldExportSpan } from "./utils";

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
  private readonly aggregateManager = new TraceAggregateManager();

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
    this.aggregateManager.onStart(span);
    super.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    this.aggregateManager.onEnd(span);

    if (
      shouldExportSpan({
        span,
        spanFilter: this.spanFilter,
      })
    ) {
      super.onEnd(span);
    }
  }

  async shutdown(): Promise<void> {
    this.aggregateManager.clear();
    return super.shutdown();
  }

  async forceFlush(): Promise<void> {
    this.aggregateManager.clear();
    return super.forceFlush();
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
  private readonly aggregateManager = new TraceAggregateManager();

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

  onStart(span: Span, parentContext: Context): void {
    this.aggregateManager.onStart(span);
    return super.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    this.aggregateManager.onEnd(span);

    if (
      shouldExportSpan({
        span,
        spanFilter: this.spanFilter,
      })
    ) {
      super.onEnd(span);
    }
  }

  async shutdown(): Promise<void> {
    this.aggregateManager.clear();
    return super.shutdown();
  }

  async forceFlush(): Promise<void> {
    this.aggregateManager.clear();
    return super.forceFlush();
  }
}
