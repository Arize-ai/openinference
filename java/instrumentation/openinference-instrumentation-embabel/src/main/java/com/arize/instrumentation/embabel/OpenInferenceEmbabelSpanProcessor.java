package com.arize.instrumentation.embabel;

import com.arize.instrumentation.ContextAttributes;
import com.arize.instrumentation.SuppressTracing;
import com.arize.instrumentation.TraceConfig;
import com.arize.instrumentation.trace.TracedSpan;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.Scope;
import io.opentelemetry.sdk.trace.ReadWriteSpan;
import io.opentelemetry.sdk.trace.ReadableSpan;
import io.opentelemetry.sdk.trace.SpanProcessor;
import java.util.Objects;

/**
 * Copies the OpenInference context attributes set with {@link ContextAttributes} (session id, user
 * id, metadata, tags) onto every span started in that context, including the spans Embabel starts
 * for agent runs, actions, LLM and tool calls. Use it together with
 * {@link OpenInferenceEmbabelSpanExporter}.
 *
 * <p>Nothing is copied while tracing is suppressed with {@link SuppressTracing} or through
 * {@link TraceConfig#isSuppressTracing()}.
 */
public final class OpenInferenceEmbabelSpanProcessor implements SpanProcessor {

    private final TraceConfig config;

    public OpenInferenceEmbabelSpanProcessor() {
        this(TraceConfig.getDefault());
    }

    public OpenInferenceEmbabelSpanProcessor(TraceConfig config) {
        this.config = Objects.requireNonNull(config, "config must not be null");
    }

    @Override
    public void onStart(Context parentContext, ReadWriteSpan span) {
        try (Scope ignored = parentContext.makeCurrent()) {
            if (SuppressTracing.isSuppressed() || config.isSuppressTracing()) {
                return;
            }
            ContextAttributes.applyToSpan(new ContextAttributeTarget(span, config));
        }
    }

    @Override
    public boolean isStartRequired() {
        return true;
    }

    @Override
    public void onEnd(ReadableSpan span) {}

    @Override
    public boolean isEndRequired() {
        return false;
    }

    /** Lets {@link ContextAttributes#applyToSpan} write to a span this processor did not start. */
    private static final class ContextAttributeTarget extends TracedSpan {
        ContextAttributeTarget(Span span, TraceConfig config) {
            super(span, Scope.noop(), config);
        }
    }
}
