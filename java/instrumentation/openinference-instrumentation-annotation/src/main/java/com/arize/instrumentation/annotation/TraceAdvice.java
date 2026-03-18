package com.arize.instrumentation.annotation;

import com.arize.instrumentation.OITracer;
import com.arize.semconv.trace.SemanticConventions;
import java.lang.reflect.Method;
import java.util.Map;
import net.bytebuddy.asm.Advice;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TraceAdvice {

    private static final Logger log = LoggerFactory.getLogger(TraceAdvice.class);

    @Advice.OnMethodEnter(suppress = Throwable.class)
    public static TracedSpan onEnter(@Advice.Origin Method method, @Advice.AllArguments Object[] args) {
        OITracer tracer = OpenInferenceAgent.getTracer();
        if (tracer == null) return null;

        TracedSpan span = null;
        try {
            String name = resolveSpanName(method);
            SemanticConventions.OpenInferenceSpanKind kind = resolveSpanKind(method);
            span = createTypedSpan(tracer, name, kind);

            // Auto-capture input
            Map<String, Object> input = SpanHelper.buildInputMap(method, args);
            if (!input.isEmpty()) {
                span.setInput(input.size() == 1 ? input.values().iterator().next() : input);
            }

            // Apply input mappings
            applyInputMappings(method, args, span);

            // Apply tool description if present
            if (span instanceof ToolSpan toolSpan) {
                Tool annotation = method.getAnnotation(Tool.class);
                if (annotation != null && !annotation.description().isEmpty()) {
                    toolSpan.setToolDescription(annotation.description());
                }
            }

            return span;
        } catch (Exception e) {
            log.warn("Failed to create trace span for method {}", method.getName(), e);
            if (span != null) {
                span.close();
            }
            return null;
        }
    }

    @Advice.OnMethodExit(onThrowable = Throwable.class, suppress = Throwable.class)
    public static void onExit(
            @Advice.Enter TracedSpan span,
            @Advice.Origin Method method,
            @Advice.Return Object result,
            @Advice.Thrown Throwable error) {
        if (span == null) return;

        try {
            if (error != null) {
                span.setError(error);
            } else if (result != null) {
                span.setOutput(result);
                applyOutputMappings(method, result, span);
            }
        } catch (Exception e) {
            log.warn("Failed to record trace span attributes for method {}", method.getName(), e);
        } finally {
            span.close();
        }
    }

    public static String resolveSpanName(Method method) {
        Chain chain = method.getAnnotation(Chain.class);
        if (chain != null) return chain.name().isEmpty() ? method.getName() : chain.name();

        LLM llm = method.getAnnotation(LLM.class);
        if (llm != null) return llm.name().isEmpty() ? method.getName() : llm.name();

        Tool tool = method.getAnnotation(Tool.class);
        if (tool != null) return tool.name().isEmpty() ? method.getName() : tool.name();

        Agent agent = method.getAnnotation(Agent.class);
        if (agent != null) return agent.name().isEmpty() ? method.getName() : agent.name();

        Span span = method.getAnnotation(Span.class);
        if (span != null) return span.name().isEmpty() ? method.getName() : span.name();

        return method.getName();
    }

    public static SemanticConventions.OpenInferenceSpanKind resolveSpanKind(Method method) {
        if (method.isAnnotationPresent(Chain.class)) return SemanticConventions.OpenInferenceSpanKind.CHAIN;
        if (method.isAnnotationPresent(LLM.class)) return SemanticConventions.OpenInferenceSpanKind.LLM;
        if (method.isAnnotationPresent(Tool.class)) return SemanticConventions.OpenInferenceSpanKind.TOOL;
        if (method.isAnnotationPresent(Agent.class)) return SemanticConventions.OpenInferenceSpanKind.AGENT;
        Span span = method.getAnnotation(Span.class);
        if (span != null) return span.kind();
        return SemanticConventions.OpenInferenceSpanKind.CHAIN;
    }

    public static TracedSpan createTypedSpan(
            OITracer tracer, String name, SemanticConventions.OpenInferenceSpanKind kind) {
        return switch (kind) {
            case LLM -> LLMSpan.start(tracer, name);
            case TOOL -> ToolSpan.start(tracer, name);
            case AGENT -> AgentSpan.start(tracer, name);
            case RETRIEVER -> RetrievalSpan.start(tracer, name);
            case EMBEDDING -> EmbeddingSpan.start(tracer, name);
            default -> ChainSpan.start(tracer, name);
        };
    }

    public static void applyInputMappings(Method method, Object[] args, TracedSpan span) {
        SpanMapping[] mappings = getInputMappings(method);
        if (mappings == null || mappings.length == 0) return;

        var params = method.getParameters();
        for (SpanMapping mapping : mappings) {
            if (mapping.param().isEmpty()) continue;
            for (int i = 0; i < params.length; i++) {
                String paramName = params[i].isNamePresent() ? params[i].getName() : "arg" + i;
                if (paramName.equals(mapping.param())) {
                    span.setAttribute(mapping.attribute(), args[i]);
                    break;
                }
            }
        }
    }

    public static void applyOutputMappings(Method method, Object result, TracedSpan span) {
        SpanMapping[] mappings = getOutputMappings(method);
        if (mappings == null || mappings.length == 0) return;

        for (SpanMapping mapping : mappings) {
            if (mapping.field().isEmpty()) continue;
            Object value = SpanHelper.extractField(result, mapping.field());
            if (value != null) {
                span.setAttribute(mapping.attribute(), value);
            }
        }
    }

    public static SpanMapping[] getInputMappings(Method method) {
        Chain chain = method.getAnnotation(Chain.class);
        if (chain != null) return chain.mapping();
        LLM llm = method.getAnnotation(LLM.class);
        if (llm != null) return llm.mapping();
        Tool tool = method.getAnnotation(Tool.class);
        if (tool != null) return tool.mapping();
        Agent agent = method.getAnnotation(Agent.class);
        if (agent != null) return agent.mapping();
        Span span = method.getAnnotation(Span.class);
        if (span != null) return span.mapping();
        return null;
    }

    public static SpanMapping[] getOutputMappings(Method method) {
        Chain chain = method.getAnnotation(Chain.class);
        if (chain != null) return chain.outputMapping();
        LLM llm = method.getAnnotation(LLM.class);
        if (llm != null) return llm.outputMapping();
        Tool tool = method.getAnnotation(Tool.class);
        if (tool != null) return tool.outputMapping();
        Agent agent = method.getAnnotation(Agent.class);
        if (agent != null) return agent.outputMapping();
        Span span = method.getAnnotation(Span.class);
        if (span != null) return span.outputMapping();
        return null;
    }
}
