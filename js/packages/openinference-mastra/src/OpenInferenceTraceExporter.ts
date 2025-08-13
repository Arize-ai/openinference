import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import type { ExportResult } from "@opentelemetry/core";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { addOpenInferenceAttributesToSpan } from "@arizeai/openinference-vercel/utils";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

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

  private extractUserInput(spans: ReadableSpan[]): string | undefined {
    // Look for the most recent user message from agent.getMostRecentUserMessage.result
    for (const span of spans) {
      if (span.name === "agent.getMostRecentUserMessage") {
        const result = span.attributes["agent.getMostRecentUserMessage.result"];
        if (typeof result === "string") {
          try {
            const messageData = JSON.parse(result);
            if (
              messageData.content &&
              typeof messageData.content === "string"
            ) {
              return messageData.content;
            }
          } catch {
            // Ignore parsing errors
          }
        }
      }
    }

    // Fallback: extract from agent.stream.argument.0 (conversation messages)
    for (const span of spans) {
      if (span.name === "agent.stream") {
        const argument = span.attributes["agent.stream.argument.0"];
        if (typeof argument === "string") {
          try {
            const messages = JSON.parse(argument);
            if (Array.isArray(messages)) {
              // Find the last user message
              for (let i = messages.length - 1; i >= 0; i--) {
                const message = messages[i];
                if (message.role === "user" && message.content) {
                  return typeof message.content === "string"
                    ? message.content
                    : message.content;
                }
              }
            }
          } catch {
            // Ignore parsing errors
          }
        }
      }
    }

    return undefined;
  }

  private extractAgentOutput(spans: ReadableSpan[]): string | undefined {
    // Look for output.value in any span (typically ai.streamText or similar)
    for (const span of spans) {
      const outputValue = span.attributes[SemanticConventions.OUTPUT_VALUE];
      if (typeof outputValue === "string") {
        return outputValue;
      }
    }
    return undefined;
  }

  private addInputOutputToRootSpans(spans: ReadableSpan[]): void {
    const rootSpans = spans.filter(
      (span) => span.parentSpanContext === undefined,
    );

    if (rootSpans.length === 0) return;

    const userInput = this.extractUserInput(spans);
    const agentOutput = this.extractAgentOutput(spans);

    // Add input and output to root spans
    for (const rootSpan of rootSpans) {
      if (userInput && !rootSpan.attributes[SemanticConventions.INPUT_VALUE]) {
        rootSpan.attributes[SemanticConventions.INPUT_VALUE] = userInput;
        rootSpan.attributes[SemanticConventions.INPUT_MIME_TYPE] = "text/plain";
      }

      if (
        agentOutput &&
        !rootSpan.attributes[SemanticConventions.OUTPUT_VALUE]
      ) {
        rootSpan.attributes[SemanticConventions.OUTPUT_VALUE] = agentOutput;
        rootSpan.attributes[SemanticConventions.OUTPUT_MIME_TYPE] =
          "text/plain";
      }
    }
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

    // Add user input and agent output to root spans for Phoenix session I/O
    this.addInputOutputToRootSpans(filteredSpans);

    super.export(filteredSpans, resultCallback);
  }
}
