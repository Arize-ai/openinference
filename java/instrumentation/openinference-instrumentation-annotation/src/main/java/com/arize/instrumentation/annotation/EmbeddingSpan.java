package com.arize.instrumentation.annotation;

import com.arize.instrumentation.OITracer;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Scope;
import java.util.List;
import java.util.Map;

public class EmbeddingSpan extends TracedSpan {

    private EmbeddingSpan(Span span, Scope scope, com.arize.instrumentation.TraceConfig config) {
        super(span, scope, config);
    }

    public static EmbeddingSpan start(OITracer tracer, String name) {
        Span span = startSpan(tracer, name, SemanticConventions.OpenInferenceSpanKind.EMBEDDING);
        Scope scope = span.makeCurrent();
        return new EmbeddingSpan(span, scope, tracer.getConfig());
    }

    public void setModelName(String model) {
        span.setAttribute(AttributeKey.stringKey(SemanticConventions.EMBEDDING_MODEL_NAME), model);
    }

    public void setEmbeddings(List<Map<String, Object>> embeddings) {
        if (config.isHideOutputEmbeddings()) return;
        SpanHelper.SerializedValue sv = SpanHelper.serialize(embeddings);
        if (sv != null) {
            span.setAttribute(AttributeKey.stringKey(SemanticConventions.EMBEDDING_EMBEDDINGS), sv.value());
        }
    }
}
