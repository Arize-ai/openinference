import { SpanStatusCode, Span } from "@opentelemetry/api";

import {
  MimeType,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { AgentTraceAggregator } from "./collector/agentTraceAggregator";
import { OITracer } from "@arizeai/openinference-core";
import { SpanCreator } from "./spanCreator";

export class CallbackHandler {
  private outputChunks: string[] = [];
  private readonly span: Span;
  private readonly traceAggregator: AgentTraceAggregator;
  private oiTracer: OITracer;

  /**
   * Callback handler for processing agent responses and traces.
   * @param params
   * @param params.oiTracer {OITracer} - The OpenTelemetry span to associate with the response.
   * @param params.span {Span} - The OpenTelemetry span to associate with the response.
   */
  constructor(oiTracer: OITracer, span: Span) {
    this.oiTracer = oiTracer;
    this.span = span;
    this.traceAggregator = new AgentTraceAggregator();
  }

  consumeResponse(chunk: Uint8Array) {
    const text = Buffer.from(chunk).toString("utf8");
    this.outputChunks.push(text);
  }

  consumeTrace(trace: Record<string, unknown>) {
    this.traceAggregator.collect(trace);
  }
  onComplete(): void {
    const finalOutput = this.outputChunks.join("");
    new SpanCreator(this.oiTracer).createSpans({
      parentSpan: this.span,
      traceNode: this.traceAggregator.rootNode,
    });
    this.span.setAttributes({
      [SemanticConventions.OUTPUT_VALUE]: finalOutput,
      [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
      [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
        "assistant",
      [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
        finalOutput,
    });
    this.span.setStatus({ code: SpanStatusCode.OK });
    this.span.end();
  }

  onError(error: unknown): void {
    if (error instanceof Error) {
      this.span.recordException(error);
    }
    const message =
      error &&
      typeof error === "object" &&
      "message" in error &&
      typeof error.message === "string"
        ? error.message
        : String(error);
    this.span.setStatus({ code: SpanStatusCode.ERROR, message });
    this.span.end();
  }
}
