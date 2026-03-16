package com.arize.instrumentation.langchain4j.utils;

import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Context;


public record SpanContext(Span span, Context context) {
    public SpanContext(Span span, Context context) {
        this.span = span;
        this.context = context; // Makes this span the "current" in context
    }
}