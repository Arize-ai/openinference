package com.arize.instrumentation.annotation;

import com.arize.instrumentation.OITracer;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Scope;

public class ChainSpan extends TracedSpan {

    private ChainSpan(Span span, Scope scope, com.arize.instrumentation.TraceConfig config) {
        super(span, scope, config);
    }

    public static ChainSpan start(OITracer tracer, String name) {
        Span span = startSpan(tracer, name, SemanticConventions.OpenInferenceSpanKind.CHAIN);
        Scope scope = span.makeCurrent();
        return new ChainSpan(span, scope, tracer.getConfig());
    }
}
