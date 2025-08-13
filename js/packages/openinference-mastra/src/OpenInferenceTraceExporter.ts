import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import type { ExportResult } from "@opentelemetry/core";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { addOpenInferenceAttributesToSpan } from "@arizeai/openinference-vercel/utils";

import { addOpenInferenceAttributesToMastraSpan } from "./attributes.js";
import { addOpenInferenceProjectResourceAttributeSpan } from "./utils.js";

type ConstructorArgs = {
  /**
   * A function that filters the spans to be exported.
   * If provided, the span will be exported if the function returns `true`.
   *
   * @example
   * ```ts
   * import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
   * import { isOpenInferenceSpan, OpenInferenceOTLPTraceExporter } from "@arizeai/openinference-mastra";
   * const spanFilter = (span: ReadableSpan) => {
   *   // add more span filtering logic here if desired
   *   // or just use the default isOpenInferenceSpan filter directly
   *   return isOpenInferenceSpan(span);
   * };
   * const exporter = new OpenInferenceOTLPTraceExporter({
   *   url: "...",
   *   headers: "...",
   *   spanFilter,
   * });
   * ```
   */
  spanFilter?: (span: ReadableSpan) => boolean;
} & NonNullable<ConstructorParameters<typeof OTLPTraceExporter>[0]>;

/**
 * A custom OpenTelemetry trace exporter that appends OpenInference semantic conventions to spans prior to export
 *
 * This class extends the `OTLPTraceExporter` and adds additional logic to the `export` method to augment the spans with OpenInference attributes.
 *
 * @example
 * ```ts
 * import { Mastra } from "@mastra/core/mastra";
 * import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
 * import { isOpenInferenceSpan, OpenInferenceOTLPTraceExporter } from "@arizeai/openinference-mastra";
 * const spanFilter = (span: ReadableSpan) => {
 *   // add more span filtering logic here if desired
 *   // or just use the default isOpenInferenceSpan filter directly
 *   return isOpenInferenceSpan(span);
 * };
 * const exporter = new OpenInferenceOTLPTraceExporter({
 *   apiKey: "api-key",
 *   collectorEndpoint: "http://localhost:6006/v1/traces",
 *   spanFilter,
 * });
 * const mastra = new Mastra({
 *   // ... other config
 *   telemetry: {
 *     export: {
 *       type: "custom",
 *       exporter,
 *     },
 *   },
 * })
 * ```
 */
export class OpenInferenceOTLPTraceExporter extends OTLPTraceExporter {
  private readonly spanFilter?: (span: ReadableSpan) => boolean;

  constructor({ spanFilter, ...args }: ConstructorArgs) {
    super({
      ...args,
    });
    this.spanFilter = spanFilter;
  }

  private isAgentOperation(span: ReadableSpan): boolean {
    return !!(
      span.name.startsWith("agent.") ||
      span.attributes.threadId ||
      span.attributes.componentName ||
      span.attributes.resourceId
    );
  }

  private getSpanId(span: ReadableSpan): string {
    return span.spanContext?.()?.spanId || "unknown";
  }

  private getTraceId(span: ReadableSpan): string {
    return span.spanContext?.()?.traceId || "unknown";
  }

  private addMissingAgentRootSpans(
    allSpans: ReadableSpan[],
    filteredSpans: ReadableSpan[],
  ): ReadableSpan[] {
    const filteredSpanIds = new Set(
      filteredSpans.map((s) => this.getSpanId(s)),
    );

    // Check filtered spans for agent operations
    const agentTraceIds = new Set<string>();
    for (const span of filteredSpans) {
      if (this.isAgentOperation(span)) {
        agentTraceIds.add(this.getTraceId(span));
      }
    }

    // Find missing root spans for agent traces
    const missingRoots: ReadableSpan[] = [];
    for (const span of allSpans) {
      if (
        span.parentSpanContext === undefined && // is root
        agentTraceIds.has(this.getTraceId(span)) && // is agent trace
        !filteredSpanIds.has(this.getSpanId(span)) && // not already included
        !span.name.startsWith("mastra.")
      ) {
        // not internal operation
        // Process the missing root span
        // TODO: duplicated logic with export()
        addOpenInferenceProjectResourceAttributeSpan(span);
        addOpenInferenceAttributesToSpan({
          ...span,
          instrumentationLibrary: {
            name: "@arizeai/openinference-mastra",
          },
        });
        addOpenInferenceAttributesToMastraSpan(span, true); // shouldMarkAsAgent = true
        missingRoots.push(span);
      }
    }

    return [...filteredSpans, ...missingRoots];
  }
  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ) {
    let filteredSpans = spans.map((span) => {
      // add OpenInference resource attributes to the span based on Mastra span attributes
      addOpenInferenceProjectResourceAttributeSpan(span);
      // add OpenInference attributes to the span based on Vercel span attributes
      addOpenInferenceAttributesToSpan({
        ...span,
        // backwards compatibility with older versions of sdk-trace-base
        instrumentationLibrary: {
          name: "@arizeai/openinference-mastra",
        },
      });
      // add OpenInference attributes to the span based on Mastra span attributes
      addOpenInferenceAttributesToMastraSpan(span);
      return span;
    });
    if (this.spanFilter) {
      filteredSpans = filteredSpans.filter(this.spanFilter);
    }
    // Add missing root spans for traces with agent operations
    filteredSpans = this.addMissingAgentRootSpans(spans, filteredSpans);
    super.export(filteredSpans, resultCallback);
  }
}
