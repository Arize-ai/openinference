import { SpanStatusCode, Span, diag } from "@opentelemetry/api";

import {
  MimeType,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { AgentTraceAggregator } from "./collector/agentTraceAggregator";
import { OITracer, safelyJSONStringify } from "@arizeai/openinference-core";
import { SpanCreator } from "./spanCreator";
import { getOutputAttributes } from "./attributes/attributeUtils";
import { extractRetrievedReferencesAttributes } from "./attributes/ragAttributeExtractionUtils";
import { Citation } from "@aws-sdk/client-bedrock-agent-runtime";

export class CallbackHandler {
  private outputChunks: string[] = [];
  private readonly span: Span;
  private readonly traceAggregator: AgentTraceAggregator;
  private oiTracer: OITracer;

  /**
   * Callback handler for processing agent responses and traces.
   * @param oiTracer {OITracer} - The OpenTelemetry span to associate with the response.
   * @param span {Span} - The OpenTelemetry span to associate with the response.

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

/**
 * RagCallbackHandler is responsible for handling streaming RAG (Retrieve and Generate)
 * responses and citations during Bedrock agent runtime instrumentation.
 *
 * This handler accumulates output and citation data as the RAG operation progresses,
 * and sets the appropriate OpenTelemetry span attributes upon completion or error.
 *
 * - handleCitation: Collects citation objects as they are received.
 * - onComplete: Sets output and citation attributes on the span and ends it.
 * - onError: Records exceptions and sets error status on the span.
 */
export class RagCallbackHandler {
  private output: string;
  private citations: Citation[];
  private readonly span: Span;

  /**
   * Callback handler for processing agent responses and traces.
   * @param span {Span} - The OpenTelemetry span to associate with the response.
   */
  constructor(span: Span) {
    this.span = span;
    this.output = "";
    this.citations = [];
  }

  handleOutput(output: string) {
    this.output = this.output + output;
  }
  handleCitation(citation: Record<string, unknown>) {
    if (citation && Object.keys(citation).length > 0) {
      this.citations.push(citation);
    }
  }
  onComplete(): void {
    try {
      this.span.setAttributes({
        ...getOutputAttributes(this.output),
        ...extractRetrievedReferencesAttributes(this.citations),
      });
    } catch (error: unknown) {
      diag.debug("Error in onComplete callback:", error);
    } finally {
      this.span.setStatus({ code: SpanStatusCode.OK });
      this.span.end();
    }
  }
  onError(error: unknown): void {
    if (error instanceof Error) {
      this.span.recordException(error);
    }
    const errorMessage =
      error instanceof Error
        ? error.message
        : (safelyJSONStringify(error) ?? undefined);
    this.span.setStatus({ code: SpanStatusCode.ERROR, message: errorMessage });
    this.span.end();
  }
}
