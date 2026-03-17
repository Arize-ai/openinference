package com.arize.instrumentation.langchain4j.utils;

import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Context;

/**
 * Utility record for holding a span and its associated context.
 * <p>
 * Used to manage the relationship between a span and its OpenTelemetry context.
 * </p>
 *
 * @param span    the OpenTelemetry span
 * @param context the OpenTelemetry context associated with the span
 */
public record SpanContext(Span span, Context context) {
    /**
     * Constructs a new SpanContext, making the span current in the context.
     *
     * @param span    the OpenTelemetry span
     * @param context the OpenTelemetry context
     */
    public SpanContext(Span span, Context context) {
        this.span = span;
        this.context = context; // Makes this span the "current" in context
    }
}