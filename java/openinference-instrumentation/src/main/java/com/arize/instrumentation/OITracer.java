package com.arize.instrumentation;

import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanBuilder;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Context;
import java.util.Objects;
import lombok.Getter;

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
    public SpanBuilder spanBuilder(String spanName) {
        return tracer.spanBuilder(spanName);
    }

    /**
     * Creates a span builder for an LLM operation.
     */
    public SpanBuilder llmSpanBuilder(String operationName, String modelName) {
        String spanName = operationName;
        if (modelName != null && !modelName.isEmpty()) {
            spanName = operationName + " " + modelName;
        }

        return spanBuilder(spanName)
                .setSpanKind(SpanKind.CLIENT)
                .setAttribute(SemanticConventions.OPENINFERENCE_SPAN_KIND, SemanticConventions.OpenInferenceSpanKind.LLM.getValue());
    }

    /**
     * Creates a span builder for a chain operation.
     */
    public SpanBuilder chainSpanBuilder(String chainName) {
        return spanBuilder(chainName)
                .setSpanKind(SpanKind.INTERNAL)
                .setAttribute(SemanticConventions.OPENINFERENCE_SPAN_KIND, SemanticConventions.OpenInferenceSpanKind.CHAIN.getValue());
    }

    /**
     * Creates a span builder for a tool operation.
     */
    public SpanBuilder toolSpanBuilder(String toolName) {
        return spanBuilder(toolName)
                .setSpanKind(SpanKind.INTERNAL)
                .setAttribute(SemanticConventions.OPENINFERENCE_SPAN_KIND, SemanticConventions.OpenInferenceSpanKind.TOOL.getValue());
    }

    /**
     * Creates a span builder for an agent operation.
     */
    public SpanBuilder agentSpanBuilder(String agentName) {
        return spanBuilder(agentName)
                .setSpanKind(SpanKind.INTERNAL)
                .setAttribute(SemanticConventions.OPENINFERENCE_SPAN_KIND, SemanticConventions.OpenInferenceSpanKind.AGENT.getValue());
    }

    /**
     * Creates a span builder for a retriever operation.
     */
    public SpanBuilder retrieverSpanBuilder(String retrieverName) {
        return spanBuilder(retrieverName)
                .setSpanKind(SpanKind.INTERNAL)
                .setAttribute(SemanticConventions.OPENINFERENCE_SPAN_KIND, SemanticConventions.OpenInferenceSpanKind.RETRIEVER.getValue());
    }

    /**
     * Creates a span builder for an embedding operation.
     */
    public SpanBuilder embeddingSpanBuilder(String operationName, String modelName) {
        String spanName = operationName;
        if (modelName != null && !modelName.isEmpty()) {
            spanName = operationName + " " + modelName;
        }

        return spanBuilder(spanName)
                .setSpanKind(SpanKind.CLIENT)
                .setAttribute(SemanticConventions.OPENINFERENCE_SPAN_KIND, SemanticConventions.OpenInferenceSpanKind.EMBEDDING.getValue());
    }

    /**
     * Creates a span builder for a reranker operation.
     */
    public SpanBuilder rerankerSpanBuilder(String rerankerName) {
        return spanBuilder(rerankerName)
                .setSpanKind(SpanKind.INTERNAL)
                .setAttribute(SemanticConventions.OPENINFERENCE_SPAN_KIND, SemanticConventions.OpenInferenceSpanKind.RERANKER.getValue());
    }

    /**
     * Creates a span builder for a guardrail operation.
     */
    public SpanBuilder guardrailSpanBuilder(String guardrailName) {
        return spanBuilder(guardrailName)
                .setSpanKind(SpanKind.INTERNAL)
                .setAttribute(SemanticConventions.OPENINFERENCE_SPAN_KIND, SemanticConventions.OpenInferenceSpanKind.GUARDRAIL.getValue());
    }

    /**
     * Creates a span builder for an evaluator operation.
     */
    public SpanBuilder evaluatorSpanBuilder(String evaluatorName) {
        return spanBuilder(evaluatorName)
                .setSpanKind(SpanKind.INTERNAL)
                .setAttribute(SemanticConventions.OPENINFERENCE_SPAN_KIND, SemanticConventions.OpenInferenceSpanKind.EVALUATOR.getValue());
    }

    /**
     * Gets the current span from the context.
     */
    public Span getCurrentSpan() {
        return Span.current();
    }

    /**
     * Gets the current span from the given context.
     */
    public Span getSpan(Context context) {
        return Span.fromContext(context);
    }
}
