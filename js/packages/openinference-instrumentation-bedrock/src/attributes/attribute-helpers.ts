import { Span, AttributeValue } from "@opentelemetry/api";

/**
 * Sets a span attribute only if the value is not null or undefined
 * Matches Python's _set_span_attribute pattern
 */
export function setSpanAttribute(span: Span, key: string, value: AttributeValue | undefined): void {
  if (value !== undefined && value !== null) {
    span.setAttribute(key, value);
  }
}

/**
 * Sets multiple span attributes with null checking
 */
export function setSpanAttributes(span: Span, attributes: Record<string, AttributeValue | undefined>): void {
  Object.entries(attributes).forEach(([key, value]) => {
    setSpanAttribute(span, key, value);
  });
}