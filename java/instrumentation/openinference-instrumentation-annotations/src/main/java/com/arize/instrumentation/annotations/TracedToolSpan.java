package com.arize.instrumentation.annotations;

import com.arize.instrumentation.OITracer;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Scope;
import java.util.Map;

public class TracedToolSpan extends TracedSpan {

    private TracedToolSpan(Span span, Scope scope, com.arize.instrumentation.TraceConfig config) {
        super(span, scope, config);
    }

    public static TracedToolSpan start(OITracer tracer, String name) {
        Span span = startSpan(tracer, name, SemanticConventions.OpenInferenceSpanKind.TOOL);
        Scope scope = span.makeCurrent();
        return new TracedToolSpan(span, scope, tracer.getConfig());
    }

    public void setToolName(String name) {
        span.setAttribute(AttributeKey.stringKey(SemanticConventions.TOOL_NAME), name);
    }

    public void setToolDescription(String description) {
        span.setAttribute(AttributeKey.stringKey(SemanticConventions.TOOL_DESCRIPTION), description);
    }

    public void setToolParameters(Map<String, Object> params) {
        if (config.isHideToolParameters()) return;
        SpanHelper.SerializedValue sv = SpanHelper.serialize(params);
        if (sv != null) {
            span.setAttribute(AttributeKey.stringKey(SemanticConventions.TOOL_PARAMETERS), sv.value());
        }
    }

    public void setToolJsonSchema(String schema) {
        span.setAttribute(AttributeKey.stringKey(SemanticConventions.TOOL_JSON_SCHEMA), schema);
    }
}
