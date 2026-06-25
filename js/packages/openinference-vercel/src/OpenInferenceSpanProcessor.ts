import type { Context } from "@opentelemetry/api";
import type { BufferConfig, ReadableSpan, Span, SpanExporter } from "@opentelemetry/sdk-trace-base";
import { BatchSpanProcessor, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";

import { TraceAggregateManager } from "./TraceAggregateManager";
import type { PreProcessSpan, SpanFilter } from "./types";
import { propagateSessionFromContext, shouldExportSpan } from "./utils";

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
 *
 * @example Agent trace mode (Eve and other AI SDK agent frameworks)
 * ```typescript
 * const processor = new OpenInferenceSimpleSpanProcessor({
 *   exporter,
 *   spanFilter: isOpenInferenceSpan,
 *   agentTraceMode: true,
 * });
 * ```
 */
export class OpenInferenceSimpleSpanProcessor extends SimpleSpanProcessor {
  private readonly spanFilter?: SpanFilter;
  private readonly agentTraceMode: boolean;
  private readonly preProcessSpan?: PreProcessSpan;
  protected readonly aggregateManager: TraceAggregateManager;

  constructor({
    exporter,
    spanFilter,
    agentTraceMode = false,
    preProcessSpan,
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
     * When true, produce a clean agent trace tree: promote the first Vercel AI
     * SDK span (`ai.*`, including framework wrappers such as Eve's `ai.eve.turn`)
     * in each trace to the trace root, label that root `AGENT` if it has no span
     * kind, propagate `session.id` from the active context, and stamp the
     * trace's earliest input / latest output onto the root when it has none of
     * its own. Defaults to `false` (no change to trace topology).
     */
    readonly agentTraceMode?: boolean;
    /**
     * A hook invoked on each span at `onEnd` before OpenInference attribute
     * conversion runs. Use it to enrich or remap framework-specific attributes.
     */
    readonly preProcessSpan?: PreProcessSpan;

    config?: BufferConfig;
  }) {
    super(exporter);
    this.spanFilter = spanFilter;
    this.agentTraceMode = agentTraceMode;
    this.preProcessSpan = preProcessSpan;
    this.aggregateManager = new TraceAggregateManager({ agentTraceMode });
  }

  onStart(span: Span, parentContext: Context): void {
    this.aggregateManager.onStart(span);
    if (this.agentTraceMode) {
      propagateSessionFromContext(span, parentContext);
    }
    super.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    this.preProcessSpan?.(span);
    // In agentTraceMode the manager may defer the promoted root and return it
    // alongside a later span, so export every span it hands back.
    for (const toExport of this.aggregateManager.onEnd(span)) {
      if (shouldExportSpan({ span: toExport, spanFilter: this.spanFilter })) {
        super.onEnd(toExport);
      }
    }
  }

  private exportPendingRoots(): void {
    for (const root of this.aggregateManager.drainPendingRoots()) {
      if (shouldExportSpan({ span: root, spanFilter: this.spanFilter })) {
        super.onEnd(root);
      }
    }
  }

  async shutdown(): Promise<void> {
    this.exportPendingRoots();
    this.aggregateManager.clear();
    return super.shutdown();
  }

  async forceFlush(): Promise<void> {
    this.exportPendingRoots();
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
 *
 * @example Agent trace mode (Eve and other AI SDK agent frameworks)
 * ```typescript
 * const processor = new OpenInferenceBatchSpanProcessor({
 *   exporter,
 *   spanFilter: isOpenInferenceSpan,
 *   agentTraceMode: true,
 * });
 * ```
 */
export class OpenInferenceBatchSpanProcessor extends BatchSpanProcessor {
  private readonly spanFilter?: SpanFilter;
  private readonly agentTraceMode: boolean;
  private readonly preProcessSpan?: PreProcessSpan;
  protected readonly aggregateManager: TraceAggregateManager;

  constructor({
    exporter,
    spanFilter,
    agentTraceMode = false,
    preProcessSpan,
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
     * When true, produce a clean agent trace tree: promote the first Vercel AI
     * SDK span (`ai.*`, including framework wrappers such as Eve's `ai.eve.turn`)
     * in each trace to the trace root, label that root `AGENT` if it has no span
     * kind, propagate `session.id` from the active context, and stamp the
     * trace's earliest input / latest output onto the root when it has none of
     * its own. Defaults to `false` (no change to trace topology).
     */
    readonly agentTraceMode?: boolean;
    /**
     * A hook invoked on each span at `onEnd` before OpenInference attribute
     * conversion runs. Use it to enrich or remap framework-specific attributes.
     */
    readonly preProcessSpan?: PreProcessSpan;
    /**
     * The configuration options for processor.
     */
    config?: BufferConfig;
  }) {
    super(exporter, config);
    this.spanFilter = spanFilter;
    this.agentTraceMode = agentTraceMode;
    this.preProcessSpan = preProcessSpan;
    this.aggregateManager = new TraceAggregateManager({ agentTraceMode });
  }

  onStart(span: Span, parentContext: Context): void {
    this.aggregateManager.onStart(span);
    if (this.agentTraceMode) {
      propagateSessionFromContext(span, parentContext);
    }
    return super.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    this.preProcessSpan?.(span);
    // In agentTraceMode the manager may defer the promoted root and return it
    // alongside a later span, so export every span it hands back.
    for (const toExport of this.aggregateManager.onEnd(span)) {
      if (shouldExportSpan({ span: toExport, spanFilter: this.spanFilter })) {
        super.onEnd(toExport);
      }
    }
  }

  private exportPendingRoots(): void {
    for (const root of this.aggregateManager.drainPendingRoots()) {
      if (shouldExportSpan({ span: root, spanFilter: this.spanFilter })) {
        super.onEnd(root);
      }
    }
  }

  async shutdown(): Promise<void> {
    this.exportPendingRoots();
    this.aggregateManager.clear();
    return super.shutdown();
  }

  async forceFlush(): Promise<void> {
    this.exportPendingRoots();
    this.aggregateManager.clear();
    return super.forceFlush();
  }
}
