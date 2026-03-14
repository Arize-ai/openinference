package com.arize.instrumentation.annotations;

import com.arize.instrumentation.OITracer;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Scope;

public class TracedAgentSpan extends TracedSpan {

    private TracedAgentSpan(Span span, Scope scope, com.arize.instrumentation.TraceConfig config) {
        super(span, scope, config);
    }

    public static TracedAgentSpan start(OITracer tracer, String name) {
        Span span = startSpan(tracer, name, SemanticConventions.OpenInferenceSpanKind.AGENT);
        Scope scope = span.makeCurrent();
        return new TracedAgentSpan(span, scope, tracer.getConfig());
    }

    public void setAgentName(String name) {
        span.setAttribute(AttributeKey.stringKey(SemanticConventions.AGENT_NAME), name);
    }
}
