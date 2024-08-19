import { ReadableSpan, SpanProcessor } from "@opentelemetry/sdk-trace-base";
import {
  getOISpanKindFromAttributes,
  getOpenInferenceAttributes,
} from "./utils";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";
import { ReadWriteSpan } from "./types";

export class OpenInferenceVercelSpanProcessor implements SpanProcessor {
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
    const spanKind = getOISpanKindFromAttributes(initialAttributes);

    (span as ReadWriteSpan).attributes = {
      ...span.attributes,
      ...getOpenInferenceAttributes({ initialAttributes, spanKind }),
      [SemanticConventions.OPENINFERENCE_SPAN_KIND]: spanKind,
    };
  }
}
