package com.arize.instrumentation;

import io.opentelemetry.context.Context;
import io.opentelemetry.context.ContextKey;
import io.opentelemetry.context.Scope;

/**
 * Utility for suppressing OpenInference tracing within a scope.
 *
 * <p>Usage:
 * <pre>{@code
 * try (Scope ignored = SuppressTracing.begin()) {
 *     // No OpenInference spans will be created in this block
 * }
 * }</pre>
 */
public final class SuppressTracing {

    static final ContextKey<Boolean> SUPPRESS_INSTRUMENTATION_KEY =
            ContextKey.named("openinference-suppress-instrumentation");

    private SuppressTracing() {}

    /**
     * Begins a scope in which OpenInference tracing is suppressed.
     * Close the returned {@link Scope} to restore normal tracing.
     */
    public static Scope begin() {
        return Context.current().with(SUPPRESS_INSTRUMENTATION_KEY, Boolean.TRUE).makeCurrent();
    }

    /**
     * Returns {@code true} if tracing is currently suppressed.
     */
    public static boolean isSuppressed() {
        return Boolean.TRUE.equals(Context.current().get(SUPPRESS_INSTRUMENTATION_KEY));
    }
}
