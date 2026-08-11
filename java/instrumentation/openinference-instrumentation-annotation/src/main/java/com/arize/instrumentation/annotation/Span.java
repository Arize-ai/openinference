package com.arize.instrumentation.annotation;

import com.arize.semconv.trace.SemanticConventions;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Base annotation for tracing methods with any OpenInference span kind.
 * Use this for less common span kinds (RETRIEVER, EMBEDDING, RERANKER, etc.)
 * that don't have a dedicated annotation.
 *
 * <p>For common span kinds, prefer the typed annotations:
 * {@link Chain}, {@link LLM}, {@link Tool}, {@link Agent}.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface Span {
    SemanticConventions.OpenInferenceSpanKind kind();

    String name() default "";

    SpanMapping[] mapping() default {};

    SpanMapping[] outputMapping() default {};
}
