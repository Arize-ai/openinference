package com.arize.instrumentation;

import io.opentelemetry.api.trace.SpanBuilder;
import io.opentelemetry.api.trace.Tracer;
import java.util.Objects;
import lombok.Getter;
import lombok.NonNull;

/**
 * OpenInference tracer wrapper that provides convenience methods for creating spans
 * with OpenInference semantic conventions.
 */
@Getter
public class OITracer implements Tracer {

    private final Tracer tracer;
    private final TraceConfig config;

    public OITracer(Tracer tracer) {
        this(tracer, TraceConfig.getDefault());
    }

    public OITracer(Tracer tracer, TraceConfig config) {
        this.tracer = Objects.requireNonNull(tracer, "tracer must not be null");
        this.config = Objects.requireNonNull(config, "config must not be null");
    }

    /**
     * Creates a span builder with the given name.
     */
    public SpanBuilder spanBuilder(@NonNull String spanName) {
        return tracer.spanBuilder(spanName);
    }
}
