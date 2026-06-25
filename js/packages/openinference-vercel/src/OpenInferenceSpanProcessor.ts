import type { Context } from "@opentelemetry/api";
import type { BufferConfig, ReadableSpan, Span, SpanExporter } from "@opentelemetry/sdk-trace-base";
import { BatchSpanProcessor, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";

import { propagateContextAttributesToSpan } from "./contextPropagation";
import { promoteReparentedRoot, reparentOrphanedSpan } from "./reparenting";
import { TraceAggregateManager } from "./TraceAggregateManager";
import type { SpanFilter } from "./types";
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
  private readonly reparentOrphanedSpans: boolean;
  private readonly propagateContextAttributes: boolean;
  private readonly aggregateManager: TraceAggregateManager;

  constructor({
    exporter,
    spanFilter,
    reparentOrphanedSpans = false,
    propagateContextAttributes = false,
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
     * Reparents AI spans that would be orphaned by span filtering. When a `spanFilter`
     * drops non-OpenInference spans (e.g. {@link isOpenInferenceSpan}), the highest-level AI
     * span (e.g. `ai.generateText`, `ai.streamText`) is often parented under a non-AI span —
     * such as the HTTP/server span Next.js parents everything under — which the filter
     * removes, leaving the AI span pointing at a parent that was never exported.
     *
     * When enabled, any AI span whose direct parent is a non-AI span is detached (re-rooted)
     * so it becomes a trace root. Handles multiple sibling AI spans per trace; AI spans nested
     * under an AI parent are left intact. The check is stateless — the parent span is read from
     * the start-time context.
     *
     * If a re-rooted root is an AI-like span the kind map doesn't recognize (e.g. a framework
     * "turn"/wrapper span), it would otherwise be kind-less and dropped by the filter; it is
     * tagged `openinference.span.kind = AGENT` so it is kept as the trace root. Recognized
     * spans keep the kind the map gave them (e.g. `ai.embed` stays CHAIN); only kind-less
     * AI-like roots get the AGENT fallback.
     *
     * Intended for use alongside a filter that drops non-AI parent spans — without such a
     * filter the non-AI parent is still exported and reparenting would split the trace.
     *
     * @default false
     */
    readonly reparentOrphanedSpans?: boolean;
    /**
     * Copies OpenInference context attributes onto each span at start time.
     *
     * The Vercel AI SDK creates its own spans, so this processor never reads the
     * OpenInference context the way the OpenInference instrumentors do. As a result,
     * values set with the `@arizeai/openinference-core` helpers — `setSession`,
     * `setUser`, `setMetadata`, `setTags` — never reach the exported AI spans.
     *
     * When enabled, every OpenInference attribute (`session.id`, `user.id`, `metadata.*`,
     * `tag.tags`, …) present on the start-time context is written directly onto the span,
     * so e.g. a `session.id` set via
     * `context.with(setSession(context.active(), { sessionId }), () => streamText(...))`
     * groups the trace into a session in Arize / Phoenix. Setting them at start time means
     * children started in the same context inherit them, and the values survive
     * {@link reparentOrphanedSpans} and export.
     *
     * @default false
     */
    readonly propagateContextAttributes?: boolean;

    config?: BufferConfig;
  }) {
    super(exporter);
    this.spanFilter = spanFilter;
    this.reparentOrphanedSpans = reparentOrphanedSpans;
    this.propagateContextAttributes = propagateContextAttributes;
    this.aggregateManager = new TraceAggregateManager();
  }

  onStart(span: Span, parentContext: Context): void {
    if (this.propagateContextAttributes) {
      propagateContextAttributesToSpan(span, parentContext);
    }
    if (this.reparentOrphanedSpans) {
      reparentOrphanedSpan(span, parentContext);
    }
    this.aggregateManager.onStart(span);
    super.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    this.aggregateManager.onEnd(span);

    if (this.reparentOrphanedSpans) {
      // A re-rooted AI wrapper the kind map didn't recognize is still kind-less here;
      // give it a fallback kind so it survives the filter as the trace root.
      promoteReparentedRoot(span);
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
  private readonly reparentOrphanedSpans: boolean;
  private readonly propagateContextAttributes: boolean;
  private readonly aggregateManager: TraceAggregateManager;

  constructor({
    exporter,
    spanFilter,
    reparentOrphanedSpans = false,
    propagateContextAttributes = false,
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
     * Reparents AI spans that would be orphaned by span filtering. When a `spanFilter`
     * drops non-OpenInference spans (e.g. {@link isOpenInferenceSpan}), the highest-level AI
     * span (e.g. `ai.generateText`, `ai.streamText`) is often parented under a non-AI span —
     * such as the HTTP/server span Next.js parents everything under — which the filter
     * removes, leaving the AI span pointing at a parent that was never exported.
     *
     * When enabled, any AI span whose direct parent is a non-AI span is detached (re-rooted)
     * so it becomes a trace root. Handles multiple sibling AI spans per trace; AI spans nested
     * under an AI parent are left intact. The check is stateless — the parent span is read from
     * the start-time context.
     *
     * If a re-rooted root is an AI-like span the kind map doesn't recognize (e.g. a framework
     * "turn"/wrapper span), it would otherwise be kind-less and dropped by the filter; it is
     * tagged `openinference.span.kind = AGENT` so it is kept as the trace root. Recognized
     * spans keep the kind the map gave them (e.g. `ai.embed` stays CHAIN); only kind-less
     * AI-like roots get the AGENT fallback.
     *
     * Intended for use alongside a filter that drops non-AI parent spans — without such a
     * filter the non-AI parent is still exported and reparenting would split the trace.
     *
     * @default false
     */
    readonly reparentOrphanedSpans?: boolean;
    /**
     * Copies OpenInference context attributes onto each span at start time.
     *
     * The Vercel AI SDK creates its own spans, so this processor never reads the
     * OpenInference context the way the OpenInference instrumentors do. As a result,
     * values set with the `@arizeai/openinference-core` helpers — `setSession`,
     * `setUser`, `setMetadata`, `setTags` — never reach the exported AI spans.
     *
     * When enabled, every OpenInference attribute (`session.id`, `user.id`, `metadata.*`,
     * `tag.tags`, …) present on the start-time context is written directly onto the span,
     * so e.g. a `session.id` set via
     * `context.with(setSession(context.active(), { sessionId }), () => streamText(...))`
     * groups the trace into a session in Arize / Phoenix. Setting them at start time means
     * children started in the same context inherit them, and the values survive
     * {@link reparentOrphanedSpans} and export.
     *
     * @default false
     */
    readonly propagateContextAttributes?: boolean;
    /**
     * The configuration options for processor.
     */
    config?: BufferConfig;
  }) {
    super(exporter, config);
    this.spanFilter = spanFilter;
    this.reparentOrphanedSpans = reparentOrphanedSpans;
    this.propagateContextAttributes = propagateContextAttributes;
    this.aggregateManager = new TraceAggregateManager();
  }

  onStart(span: Span, parentContext: Context): void {
    if (this.propagateContextAttributes) {
      propagateContextAttributesToSpan(span, parentContext);
    }
    if (this.reparentOrphanedSpans) {
      reparentOrphanedSpan(span, parentContext);
    }
    this.aggregateManager.onStart(span);
    return super.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    this.aggregateManager.onEnd(span);

    if (this.reparentOrphanedSpans) {
      // A re-rooted AI wrapper the kind map didn't recognize is still kind-less here;
      // give it a fallback kind so it survives the filter as the trace root.
      promoteReparentedRoot(span);
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

  async shutdown(): Promise<void> {
    this.aggregateManager.clear();
    return super.shutdown();
  }

  async forceFlush(): Promise<void> {
    this.aggregateManager.clear();
    return super.forceFlush();
  }
}
