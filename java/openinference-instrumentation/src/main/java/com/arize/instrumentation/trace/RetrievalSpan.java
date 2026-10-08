package com.arize.instrumentation.trace;

import com.arize.instrumentation.OITracer;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Scope;
import java.util.List;
import java.util.Map;

public class RetrievalSpan extends TracedSpan {

    private RetrievalSpan(Span span, Scope scope, com.arize.instrumentation.TraceConfig config) {
        super(span, scope, config);
    }

    public static RetrievalSpan start(OITracer tracer, String name) {
        Span span = startSpan(tracer, name, SemanticConventions.OpenInferenceSpanKind.RETRIEVER);
        Scope scope = span.makeCurrent();
        return new RetrievalSpan(span, scope, tracer.getConfig());
    }

    public void setDocuments(List<Map<String, Object>> docs) {
        SpanSerializer.SerializedValue sv = SpanSerializer.serialize(docs);
        if (sv != null) {
            span.setAttribute(AttributeKey.stringKey(SemanticConventions.RETRIEVAL_DOCUMENTS), sv.value());
        }
    }
}
