package com.arize.instrumentation.trace;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.TraceConfig;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Scope;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public abstract class TracedSpan implements AutoCloseable {

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
        SpanSerializer.SerializedValue sv = SpanSerializer.serialize(metadata);
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
        if (key == null || key.isEmpty() || value == null) {
            return;
        }
        if (SemanticConventions.INPUT_VALUE.equals(key) && config.isHideInputs()) {
            return;
        }
        if (SemanticConventions.OUTPUT_VALUE.equals(key) && config.isHideOutputs()) {
            return;
        }

        if (value instanceof String s) {
            span.setAttribute(AttributeKey.stringKey(key), s);
            return;
        }
        if (value instanceof Boolean b) {
            span.setAttribute(AttributeKey.booleanKey(key), b);
            return;
        }
        if (value instanceof Integer i) {
            span.setAttribute(AttributeKey.longKey(key), i.longValue());
            return;
        }
        if (value instanceof Long l) {
            span.setAttribute(AttributeKey.longKey(key), l);
            return;
        }
        if (value instanceof Short s) {
            span.setAttribute(AttributeKey.longKey(key), s.longValue());
            return;
        }
        if (value instanceof Byte b) {
            span.setAttribute(AttributeKey.longKey(key), b.longValue());
            return;
        }
        if (value instanceof Double d) {
            span.setAttribute(AttributeKey.doubleKey(key), d);
            return;
        }
        if (value instanceof Float f) {
            span.setAttribute(AttributeKey.doubleKey(key), f.doubleValue());
            return;
        }
        if (value instanceof Number n) {
            span.setAttribute(AttributeKey.doubleKey(key), n.doubleValue());
            return;
        }
        if (value instanceof List<?> list && canSerializeStringCollection(list)) {
            span.setAttribute(AttributeKey.stringArrayKey(key), toStringList(list));
            return;
        }
        if (value instanceof String[] arr) {
            span.setAttribute(AttributeKey.stringArrayKey(key), List.of(arr));
            return;
        }
        if (value instanceof Iterable<?> iterable) {
            List<String> converted = tryConvertIterable(iterable);
            if (converted != null) {
                span.setAttribute(AttributeKey.stringArrayKey(key), converted);
                return;
            }
        }

        SpanSerializer.SerializedValue sv = SpanSerializer.serialize(value);
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
        SpanSerializer.SerializedValue sv = SpanSerializer.serialize(value);
        if (sv == null) return;
        span.setAttribute(AttributeKey.stringKey(valueKey), sv.value());
        String mimeType = sv.isJson()
                ? SemanticConventions.MimeType.JSON.getValue()
                : SemanticConventions.MimeType.TEXT.getValue();
        span.setAttribute(AttributeKey.stringKey(mimeKey), mimeType);
    }

    private static boolean canSerializeStringCollection(List<?> list) {
        for (Object item : list) {
            if (!(item instanceof String)) {
                return false;
            }
        }
        return true;
    }

    private static List<String> toStringList(List<?> list) {
        return list.stream().map(obj -> (String) obj).toList();
    }

    private static List<String> tryConvertIterable(Iterable<?> source) {
        List<String> values = new ArrayList<>();
        for (Object item : source) {
            if (!(item instanceof String s)) {
                return null;
            }
            values.add(s);
        }
        return values;
    }
}
