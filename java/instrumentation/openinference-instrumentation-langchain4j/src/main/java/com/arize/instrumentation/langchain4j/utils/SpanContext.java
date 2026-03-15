package com.arize.instrumentation.langchain4j.utils;

import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Context;

public class SpanContext {
    public final Span span;
    final Context context;

    public SpanContext(Span span, Context context) {
        this.span = span;
        this.context = context;
    }

}