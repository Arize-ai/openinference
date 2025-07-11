package com.arize.instrumentation.trace.config;

import com.arize.instrumentation.TraceConfig;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.common.Value;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanContext;
import io.opentelemetry.api.trace.StatusCode;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;

import java.util.concurrent.TimeUnit;

import static com.arize.instrumentation.trace.config.MaskingUtils.mask;

@RequiredArgsConstructor
public class OISpan implements Span {
    @NonNull
    private Span span;
    @NonNull
    private TraceConfig config;

    @Override
    public <T> Span setAttribute(AttributeKey<T> key, T value) {
        Value<T> maskedValue = mask(config, key.getKey(), (Value<T>) value);
        if (maskedValue != null) {
            span.setAttribute(key, maskedValue.getValue());
        }
        return span;
    }

    @Override
    public Span addEvent(String name, Attributes attributes) {
        span.addEvent(name, attributes);
        return span;
    }

    @Override
    public Span addEvent(String name, Attributes attributes, long timestamp, TimeUnit unit) {
        span.addEvent(name, attributes, timestamp, unit);
        return span;
    }

    @Override
    public Span setStatus(StatusCode statusCode, String description) {
        span.setStatus(statusCode, description);
        return span;
    }

    @Override
    public Span recordException(Throwable exception, Attributes additionalAttributes) {
        span.recordException(exception, additionalAttributes);
        return span;
    }

    @Override
    public Span updateName(String name) {
        span.updateName(name);
        return span;
    }

    @Override
    public void end() {
        span.end();
    }

    @Override
    public void end(long timestamp, TimeUnit unit) {
        span.end(timestamp, unit);
    }

    @Override
    public SpanContext getSpanContext() {
        return span.getSpanContext();
    }

    @Override
    public boolean isRecording() {
        return span.isRecording();
    }
}
