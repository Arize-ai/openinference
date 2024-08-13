import { ReadableSpan, SpanProcessor } from "@opentelemetry/sdk-trace-base";
import { hasAIAttributes } from "./utils";

export class OpenInferenceSpanProcessor implements SpanProcessor {
  async forceFlush() {
    // No-op
  }

  onStart(_: ReadableSpan): void {
    // No-op
  }

  async shutdown(): Promise<void> {
    // No-op
  }
  onEnd(span: ReadableSpan): void {
    const initialAttributes = span.attributes;
    if (!hasAIAttributes(initialAttributes)) {
      return;
    }
  }
}
