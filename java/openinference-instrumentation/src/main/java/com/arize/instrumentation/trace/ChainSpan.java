package com.arize.instrumentation.trace;

import com.arize.instrumentation.OITracer;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Scope;

public class ChainSpan extends TracedSpan {

    private ChainSpan(Span span, Scope scope, com.arize.instrumentation.TraceConfig config) {
        super(span, scope, config);
    }

    public static ChainSpan start(OITracer tracer, String name) {
        return start(tracer, name, SemanticConventions.OpenInferenceSpanKind.CHAIN);
    }

    /**
     * Start a span with an explicit kind. Used for span kinds (e.g. RERANKER, GUARDRAIL,
     * EVALUATOR) that don't have a dedicated typed span class.
     */
    public static ChainSpan start(OITracer tracer, String name, SemanticConventions.OpenInferenceSpanKind kind) {
        Span span = startSpan(tracer, name, kind);
        Scope scope = span.makeCurrent();
        return new ChainSpan(span, scope, tracer.getConfig());
    }
}
