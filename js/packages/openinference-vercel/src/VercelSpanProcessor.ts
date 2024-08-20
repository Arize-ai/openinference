import { ReadableSpan, SpanProcessor } from "@opentelemetry/sdk-trace-base";
import {
  safelyGetOISpanKindFromAttributes,
  safelyGetOpenInferenceAttributes,
} from "./utils";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";
import { ReadWriteSpan } from "./types";

export class VercelSpanProcessor implements SpanProcessor {
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
    const initialAttributes = { ...span.attributes };
    const spanKind = safelyGetOISpanKindFromAttributes(initialAttributes);

    (span as ReadWriteSpan).attributes = {
      ...span.attributes,
      ...safelyGetOpenInferenceAttributes({ initialAttributes, spanKind }),
      [SemanticConventions.OPENINFERENCE_SPAN_KIND]: spanKind ?? undefined,
    };
  }
}
