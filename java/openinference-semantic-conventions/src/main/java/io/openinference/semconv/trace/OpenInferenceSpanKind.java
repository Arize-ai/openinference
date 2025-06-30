package io.openinference.semconv.trace;

/**
 * OpenInference span kind values.
 */
public enum OpenInferenceSpanKind {
    TOOL("TOOL"),
    CHAIN("CHAIN"),
    LLM("LLM"),
    RETRIEVER("RETRIEVER"),
    EMBEDDING("EMBEDDING"),
    AGENT("AGENT"),
    RERANKER("RERANKER"),
    UNKNOWN("UNKNOWN"),
    GUARDRAIL("GUARDRAIL"),
    EVALUATOR("EVALUATOR");
    
    private final String value;
    
    OpenInferenceSpanKind(String value) {
        this.value = value;
    }
    
    public String getValue() {
        return value;
    }
    
    @Override
    public String toString() {
        return value;
    }
} 