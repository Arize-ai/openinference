import {
  Span,
  SpanContext,
  SpanStatus,
  TimeInput,
  Attributes,
  Exception,
  AttributeValue,
  Link,
} from "@opentelemetry/api";
import { TraceConfig } from "./types";
import { mask } from "./maskingRules";

/**
 * A wrapper around the OpenTelemetry {@link Span} interface that masks sensitive information based on the passed in {@link TraceConfig}.
 */
export class OISpan implements Span {
  private readonly span: Span;
  private readonly config: TraceConfig;

  constructor({ span, config }: { span: Span; config: TraceConfig }) {
    this.span = span;
    this.config = config;
  }

  setAttribute(key: string, value: AttributeValue): this {
    const maskedValue = mask({ config: this.config, key, value });
    if (maskedValue != null) {
      this.span.setAttribute(key, maskedValue);
    }
    return this;
  }

  setAttributes(attributes: Attributes): this {
    const maskedAttributes = Object.entries(attributes).reduce(
      (maskedAttributes, [key, value]) => {
        maskedAttributes[key] = mask({ config: this.config, key, value });
        return maskedAttributes;
      },
      {} as Attributes,
    );
    this.span.setAttributes(maskedAttributes);
    return this;
  }

  addEvent(
    name: string,
    attributesOrStartTime?: Attributes | TimeInput,
    startTime?: TimeInput,
  ): this {
    this.span.addEvent(name, attributesOrStartTime, startTime);
    return this;
  }

  setStatus(status: SpanStatus): this {
    this.span.setStatus(status);
    return this;
  }

  updateName(name: string): this {
    this.span.updateName(name);
    return this;
  }

  end(endTime?: TimeInput): void {
    this.span.end(endTime);
  }

  isRecording(): boolean {
    return this.span.isRecording();
  }

  recordException(exception: Exception, time?: TimeInput): void {
    this.span.recordException(exception, time);
  }

  spanContext(): SpanContext {
    return this.span.spanContext();
  }

  addLink(link: Link): this {
    this.span.addLink(link);
    return this;
  }

  addLinks(links: Link[]): this {
    this.span.addLinks(links);
    return this;
  }
}
