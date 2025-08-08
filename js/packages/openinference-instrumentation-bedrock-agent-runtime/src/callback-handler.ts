import { diag, SpanStatusCode, Span } from "@opentelemetry/api";

import {
  MimeType,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

export class CallbackHandler {
  private outputChunks: string[] = [];
  private span: Span;

  constructor(span: Span) {
    this.span = span;
  }

  consumeResponse(chunk: Uint8Array) {
    const text = Buffer.from(chunk).toString("utf8");
    this.outputChunks.push(text);
  }

  onComplete(): void {
    const finalOutput = this.outputChunks.join("");
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
