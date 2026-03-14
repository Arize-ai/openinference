package com.arize.instrumentation.annotations;

import com.arize.instrumentation.OITracer;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Scope;

public class TracedChainSpan extends TracedSpan {

    private TracedChainSpan(Span span, Scope scope, com.arize.instrumentation.TraceConfig config) {
        super(span, scope, config);
    }

    public static TracedChainSpan start(OITracer tracer, String name) {
        Span span = startSpan(tracer, name, SemanticConventions.OpenInferenceSpanKind.CHAIN);
        Scope scope = span.makeCurrent();
        return new TracedChainSpan(span, scope, tracer.getConfig());
    }
}
