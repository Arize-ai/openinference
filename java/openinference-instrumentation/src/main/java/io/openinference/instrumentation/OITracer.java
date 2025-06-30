package io.openinference.instrumentation;

import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanBuilder;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Context;
import io.openinference.semconv.trace.OpenInferenceSpanKind;
import io.openinference.semconv.trace.SpanAttributes;

import java.util.Objects;

/**
 * OpenInference tracer wrapper that provides convenience methods for creating spans
 * with OpenInference semantic conventions.
 */
public class OITracer {
    
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
                .setAttribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.LLM.getValue());
    }
    
    /**
     * Creates a span builder for a chain operation.
     */
    public SpanBuilder chainSpanBuilder(String chainName) {
        return spanBuilder(chainName)
                .setSpanKind(SpanKind.INTERNAL)
                .setAttribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.CHAIN.getValue());
    }
    
    /**
     * Creates a span builder for a tool operation.
     */
    public SpanBuilder toolSpanBuilder(String toolName) {
        return spanBuilder(toolName)
                .setSpanKind(SpanKind.INTERNAL)
                .setAttribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.TOOL.getValue());
    }
    
    /**
     * Creates a span builder for an agent operation.
     */
    public SpanBuilder agentSpanBuilder(String agentName) {
        return spanBuilder(agentName)
                .setSpanKind(SpanKind.INTERNAL)
                .setAttribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.AGENT.getValue());
    }
    
    /**
     * Creates a span builder for a retriever operation.
     */
    public SpanBuilder retrieverSpanBuilder(String retrieverName) {
        return spanBuilder(retrieverName)
                .setSpanKind(SpanKind.INTERNAL)
                .setAttribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.RETRIEVER.getValue());
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
                .setAttribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.EMBEDDING.getValue());
    }
    
    /**
     * Creates a span builder for a reranker operation.
     */
    public SpanBuilder rerankerSpanBuilder(String rerankerName) {
        return spanBuilder(rerankerName)
                .setSpanKind(SpanKind.INTERNAL)
                .setAttribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.RERANKER.getValue());
    }
    
    /**
     * Creates a span builder for a guardrail operation.
     */
    public SpanBuilder guardrailSpanBuilder(String guardrailName) {
        return spanBuilder(guardrailName)
                .setSpanKind(SpanKind.INTERNAL)
                .setAttribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.GUARDRAIL.getValue());
    }
    
    /**
     * Creates a span builder for an evaluator operation.
     */
    public SpanBuilder evaluatorSpanBuilder(String evaluatorName) {
        return spanBuilder(evaluatorName)
                .setSpanKind(SpanKind.INTERNAL)
                .setAttribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.EVALUATOR.getValue());
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
    
    /**
     * Gets the trace configuration.
     */
    public TraceConfig getConfig() {
        return config;
    }
    
    /**
     * Gets the underlying OpenTelemetry tracer.
     */
    public Tracer getTracer() {
        return tracer;
    }
} 