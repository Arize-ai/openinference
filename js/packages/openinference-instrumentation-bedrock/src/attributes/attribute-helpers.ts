import { Span } from "@opentelemetry/api";

/**
 * Sets a span attribute only if the value is not null or undefined
 * Provides null-safe attribute setting for OpenTelemetry spans
 */
export function setSpanAttribute(span: Span, key: string, value: any) {
  if (value !== undefined && value !== null) {
    span.setAttribute(key, value);
  }
}

/**
 * Sets multiple span attributes with null checking
 */
export function setSpanAttributes(span: Span, attributes: Record<string, any>) {
  Object.entries(attributes).forEach(([key, value]) => {
    setSpanAttribute(span, key, value);
  });
}