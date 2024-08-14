import { ReadableSpan, SpanProcessor } from "@opentelemetry/sdk-trace-base";
import {
  getOIModelNameAttribute,
  getOISpanKindFromAttributes,
  hasAIAttributes,
} from "./utils";
import { Span } from "@opentelemetry/api";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

export class OpenInferenceSpanProcessor implements SpanProcessor {
  async forceFlush() {
    // No-op
  }

  onStart(_: Span): void {
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

    const spanKind = getOISpanKindFromAttributes(initialAttributes);

    if (spanKind == null) {
      return;
    }
    // @ts-expect-error - This is a read-only span and thus has no setter for attributes
    // Manually patch attributes here
    span.attributes = {
      ...span.attributes,
      ...getOIModelNameAttribute(initialAttributes),
      [SemanticConventions.OPENINFERENCE_SPAN_KIND]: spanKind,
    };
  }
}
