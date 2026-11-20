import type {
  Attributes,
  AttributeValue,
  Exception,
  Link,
  Span,
  SpanContext,
  SpanStatus,
  TimeInput,
} from "@opentelemetry/api";
import { diag } from "@opentelemetry/api";

import { isPresent } from "../../utils";
import { getGenAIAttributes } from "../genai";
import { mask } from "./maskingRules";
import type { TraceConfig } from "./types";

/**
 * A wrapper around the OpenTelemetry {@link Span} interface that masks sensitive information based on the passed in {@link TraceConfig}.
 *
 * When `enableGenAISemconv` is set on the {@link TraceConfig}, the span also emits
 * OpenTelemetry GenAI semantic convention (`gen_ai.*`) attributes derived from its
 * (masked) OpenInference attributes when it ends.
 */
export class OISpan implements Span {
  private readonly span: Span;
  private readonly config: TraceConfig;
  /**
   * The masked attributes set through this span, used to derive `gen_ai.*`
   * attributes at span end. Only tracked when `enableGenAISemconv` is on, since
   * the OpenTelemetry API offers no way to read attributes back from a span.
   */
  private readonly recordedAttributes: Attributes = {};
  /** Whether `end` has run, so the `gen_ai.*` attributes are derived only once. */
  private hasEnded = false;

  constructor({ span, config }: { span: Span; config: TraceConfig }) {
    this.span = span;
    this.config = config;
  }

  setAttribute(key: string, value: AttributeValue): this {
    const maskedValue = mask({ config: this.config, key, value });
    if (isPresent(maskedValue)) {
      this.span.setAttribute(key, maskedValue);
      this.recordAttribute({ key, value: maskedValue });
    }
    return this;
  }

  setAttributes(attributes: Attributes): this {
    const maskedAttributes = Object.entries(attributes).reduce<Attributes>((acc, [key, value]) => {
      acc[key] = mask({ config: this.config, key, value });
      return acc;
    }, {});
    this.span.setAttributes(maskedAttributes);
    for (const [key, value] of Object.entries(maskedAttributes)) {
      this.recordAttribute({ key, value });
    }
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
    if (this.config.enableGenAISemconv && !this.hasEnded) {
      this.setGenAIAttributes();
    }
    this.hasEnded = true;
    this.span.end(endTime);
  }

  /**
   * Remembers a masked attribute so the `gen_ai.*` attributes can be derived
   * from it at span end. Does nothing unless `enableGenAISemconv` is on.
   *
   * @param params.key - The attribute key
   * @param params.value - The masked attribute value
   */
  private recordAttribute({ key, value }: { key: string; value?: AttributeValue }): void {
    if (this.config.enableGenAISemconv && isPresent(value)) {
      this.recordedAttributes[key] = value;
    }
  }

  /**
   * Sets the `gen_ai.*` attributes derived from the attributes recorded on this
   * span, skipping any attribute that was already set explicitly.
   */
  private setGenAIAttributes(): void {
    if (!this.span.isRecording()) {
      return;
    }
    try {
      const genAIAttributes = getGenAIAttributes(this.recordedAttributes);
      for (const [key, value] of Object.entries(genAIAttributes)) {
        if (isPresent(value) && !(key in this.recordedAttributes)) {
          this.span.setAttribute(key, value);
        }
      }
    } catch (error) {
      diag.warn("OpenInference failed to derive GenAI semantic convention attributes", error);
    }
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
