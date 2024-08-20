import { ReadableSpan, SpanProcessor } from "@opentelemetry/sdk-trace-base";
import { safelyGetOpenInferenceAttributes } from "./utils";
import { ReadWriteSpan } from "./types";

export class OpenInferenceSpanProcessor implements SpanProcessor {
  async forceFlush(): Promise<void> {
    // no-op
  }

  onStart(): void {
    // no-op
  }

  async shutdown(): Promise<void> {
    // no-op
  }

  onEnd(span: ReadableSpan): void {
    const attributes = { ...span.attributes };

    (span as ReadWriteSpan).attributes = {
      ...span.attributes,
      ...safelyGetOpenInferenceAttributes(attributes),
    };
  }
}
