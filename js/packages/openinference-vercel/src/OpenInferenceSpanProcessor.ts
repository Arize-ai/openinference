import type { Context } from "@opentelemetry/api";
import type { BufferConfig, ReadableSpan, Span, SpanExporter } from "@opentelemetry/sdk-trace-base";
import { BatchSpanProcessor, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";

import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { TraceAggregateManager } from "./TraceAggregateManager";
import type { SpanFilter } from "./types";
import { shouldExportSpan } from "./utils";

const DEFAULT_ROOT_SPAN_TRACE_ID_CACHE_SIZE = 1000;
const VERCEL_ROOT_SPAN_NAMES = new Set(["ai.eve.turn"]);

class RootSpanManager {
  private readonly preserveVercelRootSpans: boolean;
  private readonly maxTraceIds: number;
  private readonly rootSpanIdsByTraceId = new Map<string, string>();

  constructor({
    preserveVercelRootSpans,
    rootSpanTraceIdCacheSize,
  }: {
    readonly preserveVercelRootSpans?: boolean;
    readonly rootSpanTraceIdCacheSize?: number;
  }) {
    this.preserveVercelRootSpans = preserveVercelRootSpans ?? false;
    this.maxTraceIds = Math.max(
      1,
      rootSpanTraceIdCacheSize ?? DEFAULT_ROOT_SPAN_TRACE_ID_CACHE_SIZE,
    );
  }

  onStart(span: Span): void {
    if (!this.preserveVercelRootSpans || !VERCEL_ROOT_SPAN_NAMES.has(span.name)) {
      return;
    }

    const { traceId, spanId } = span.spanContext();
    if (this.rootSpanIdsByTraceId.has(traceId)) {
      return;
    }

    // parentSpanId is readonly on the public Span type; runtime SDK spans are mutable.
    (span as unknown as { parentSpanId?: string }).parentSpanId = undefined;
    (span as unknown as { parentSpanContext?: unknown }).parentSpanContext = undefined;
    span.setAttribute(SemanticConventions.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.AGENT);
    this.rootSpanIdsByTraceId.set(traceId, spanId);
    this.enforceMaxTraceIds();
  }

  shouldExport(span: ReadableSpan): boolean {
    const { traceId, spanId } = span.spanContext();
    return this.rootSpanIdsByTraceId.get(traceId) === spanId;
  }

  clear(): void {
    this.rootSpanIdsByTraceId.clear();
  }

  private enforceMaxTraceIds(): void {
    while (this.rootSpanIdsByTraceId.size > this.maxTraceIds) {
      const oldestTraceId = this.rootSpanIdsByTraceId.keys().next().value;
      if (oldestTraceId == null) {
        return;
      }
      this.rootSpanIdsByTraceId.delete(oldestTraceId);
    }
  }
}

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
  private readonly rootSpanManager: RootSpanManager;

  constructor({
    exporter,
    spanFilter,
    preserveVercelRootSpans,
    rootSpanTraceIdCacheSize,
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
     * Whether to preserve known Vercel framework root spans when filtering would otherwise drop their parents.
     */
    readonly preserveVercelRootSpans?: boolean;
    /**
     * The maximum number of trace IDs to remember for root span promotion.
     */
    readonly rootSpanTraceIdCacheSize?: number;

    config?: BufferConfig;
  }) {
    super(exporter);
    this.spanFilter = spanFilter;
    this.rootSpanManager = new RootSpanManager({
      preserveVercelRootSpans,
      rootSpanTraceIdCacheSize,
    });
  }

  onStart(span: Span, parentContext: Context): void {
    this.rootSpanManager.onStart(span);
    this.aggregateManager.onStart(span);
    super.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    this.aggregateManager.onEnd(span);

    if (
      shouldExportSpan({
        span,
        spanFilter: this.spanFilter,
      }) ||
      this.rootSpanManager.shouldExport(span)
    ) {
      super.onEnd(span);
    }
  }

  async shutdown(): Promise<void> {
    this.aggregateManager.clear();
    this.rootSpanManager.clear();
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
  private readonly rootSpanManager: RootSpanManager;

  constructor({
    exporter,
    spanFilter,
    preserveVercelRootSpans,
    rootSpanTraceIdCacheSize,
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
     * Whether to preserve known Vercel framework root spans when filtering would otherwise drop their parents.
     */
    readonly preserveVercelRootSpans?: boolean;
    /**
     * The maximum number of trace IDs to remember for root span promotion.
     */
    readonly rootSpanTraceIdCacheSize?: number;
    /**
     * The configuration options for processor.
     */
    config?: BufferConfig;
  }) {
    super(exporter, config);
    this.spanFilter = spanFilter;
    this.rootSpanManager = new RootSpanManager({
      preserveVercelRootSpans,
      rootSpanTraceIdCacheSize,
    });
  }

  onStart(span: Span, parentContext: Context): void {
    this.rootSpanManager.onStart(span);
    this.aggregateManager.onStart(span);
    return super.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    this.aggregateManager.onEnd(span);

    if (
      shouldExportSpan({
        span,
        spanFilter: this.spanFilter,
      }) ||
      this.rootSpanManager.shouldExport(span)
    ) {
      super.onEnd(span);
    }
  }

  async shutdown(): Promise<void> {
    this.aggregateManager.clear();
    this.rootSpanManager.clear();
    return super.shutdown();
  }

  async forceFlush(): Promise<void> {
    this.aggregateManager.clear();
    return super.forceFlush();
  }
}
