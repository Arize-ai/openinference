package com.arize.instrumentation.annotations;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.TraceConfig;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Scope;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class TracedSpan implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(TracedSpan.class);

    protected final Span span;
    private final Scope scope;
    protected final TraceConfig config;
    private boolean errorRecorded = false;
    private boolean closed = false;

    protected TracedSpan(Span span, Scope scope, TraceConfig config) {
        this.span = span;
        this.scope = scope;
        this.config = config;
    }

    protected static Span startSpan(OITracer tracer, String name, SemanticConventions.OpenInferenceSpanKind kind) {
        Span span = tracer.spanBuilder(name).startSpan();
        span.setAttribute(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND), kind.getValue());
        return span;
    }

    public void setInput(Object input) {
        if (config.isHideInputs()) return;
        setValueAttribute(SemanticConventions.INPUT_VALUE, SemanticConventions.INPUT_MIME_TYPE, input);
    }

    public void setOutput(Object output) {
        if (output == null) return;
        if (config.isHideOutputs()) return;
        setValueAttribute(SemanticConventions.OUTPUT_VALUE, SemanticConventions.OUTPUT_MIME_TYPE, output);
    }

    public void setMetadata(Map<String, Object> metadata) {
        SpanHelper.SerializedValue sv = SpanHelper.serialize(metadata);
        if (sv != null) {
            span.setAttribute(AttributeKey.stringKey(SemanticConventions.METADATA), sv.value());
        }
    }

    public void setTags(List<String> tags) {
        span.setAttribute(AttributeKey.stringArrayKey(SemanticConventions.TAG_TAGS), tags);
    }

    public void setSessionId(String sessionId) {
        span.setAttribute(AttributeKey.stringKey(SemanticConventions.SESSION_ID), sessionId);
    }

    public void setUserId(String userId) {
        span.setAttribute(AttributeKey.stringKey(SemanticConventions.USER_ID), userId);
    }

    public void setAttribute(String key, Object value) {
        SpanHelper.SerializedValue sv = SpanHelper.serialize(value);
        if (sv != null) {
            span.setAttribute(AttributeKey.stringKey(key), sv.value());
        }
    }

    public void setError(Throwable t) {
        errorRecorded = true;
        span.recordException(t);
        span.setStatus(StatusCode.ERROR, t.getMessage());
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (!errorRecorded) {
            span.setStatus(StatusCode.OK);
        }
        scope.close();
        span.end();
    }

    protected void setValueAttribute(String valueKey, String mimeKey, Object value) {
        SpanHelper.SerializedValue sv = SpanHelper.serialize(value);
        if (sv == null) return;
        span.setAttribute(AttributeKey.stringKey(valueKey), sv.value());
        String mimeType = sv.isJson()
                ? SemanticConventions.MimeType.JSON.getValue()
                : SemanticConventions.MimeType.TEXT.getValue();
        span.setAttribute(AttributeKey.stringKey(mimeKey), mimeType);
    }
}
