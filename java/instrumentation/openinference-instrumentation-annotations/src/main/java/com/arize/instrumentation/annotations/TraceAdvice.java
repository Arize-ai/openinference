package com.arize.instrumentation.annotations;

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

        try {
            String name = resolveSpanName(method);
            SemanticConventions.OpenInferenceSpanKind kind = resolveSpanKind(method);
            TracedSpan span = createTypedSpan(tracer, name, kind);

            // Auto-capture input
            Map<String, Object> input = SpanHelper.buildInputMap(method, args);
            if (!input.isEmpty()) {
                span.setInput(input.size() == 1 ? input.values().iterator().next() : input);
            }

            // Apply input mappings
            applyInputMappings(method, args, span);

            // Apply tool description if present
            if (span instanceof TracedToolSpan toolSpan) {
                TraceTool annotation = method.getAnnotation(TraceTool.class);
                if (annotation != null && !annotation.description().isEmpty()) {
                    toolSpan.setToolDescription(annotation.description());
                }
            }

            return span;
        } catch (Exception e) {
            log.warn("Failed to create trace span for method {}", method.getName(), e);
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
        TraceChain chain = method.getAnnotation(TraceChain.class);
        if (chain != null) return chain.name().isEmpty() ? method.getName() : chain.name();

        TraceLLM llm = method.getAnnotation(TraceLLM.class);
        if (llm != null) return llm.name().isEmpty() ? method.getName() : llm.name();

        TraceTool tool = method.getAnnotation(TraceTool.class);
        if (tool != null) return tool.name().isEmpty() ? method.getName() : tool.name();

        TraceAgent agent = method.getAnnotation(TraceAgent.class);
        if (agent != null) return agent.name().isEmpty() ? method.getName() : agent.name();

        return method.getName();
    }

    public static SemanticConventions.OpenInferenceSpanKind resolveSpanKind(Method method) {
        if (method.isAnnotationPresent(TraceChain.class)) return SemanticConventions.OpenInferenceSpanKind.CHAIN;
        if (method.isAnnotationPresent(TraceLLM.class)) return SemanticConventions.OpenInferenceSpanKind.LLM;
        if (method.isAnnotationPresent(TraceTool.class)) return SemanticConventions.OpenInferenceSpanKind.TOOL;
        if (method.isAnnotationPresent(TraceAgent.class)) return SemanticConventions.OpenInferenceSpanKind.AGENT;
        return SemanticConventions.OpenInferenceSpanKind.CHAIN;
    }

    public static TracedSpan createTypedSpan(
            OITracer tracer, String name, SemanticConventions.OpenInferenceSpanKind kind) {
        return switch (kind) {
            case LLM -> TracedLLMSpan.start(tracer, name);
            case TOOL -> TracedToolSpan.start(tracer, name);
            case AGENT -> TracedAgentSpan.start(tracer, name);
            default -> TracedChainSpan.start(tracer, name);
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
                    span.setAttribute(mapping.attribute().getKey(), args[i]);
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
                span.setAttribute(mapping.attribute().getKey(), value);
            }
        }
    }

    public static SpanMapping[] getInputMappings(Method method) {
        TraceChain chain = method.getAnnotation(TraceChain.class);
        if (chain != null) return chain.mapping();
        TraceLLM llm = method.getAnnotation(TraceLLM.class);
        if (llm != null) return llm.mapping();
        TraceTool tool = method.getAnnotation(TraceTool.class);
        if (tool != null) return tool.mapping();
        TraceAgent agent = method.getAnnotation(TraceAgent.class);
        if (agent != null) return agent.mapping();
        return null;
    }

    public static SpanMapping[] getOutputMappings(Method method) {
        TraceChain chain = method.getAnnotation(TraceChain.class);
        if (chain != null) return chain.outputMapping();
        TraceLLM llm = method.getAnnotation(TraceLLM.class);
        if (llm != null) return llm.outputMapping();
        TraceTool tool = method.getAnnotation(TraceTool.class);
        if (tool != null) return tool.outputMapping();
        TraceAgent agent = method.getAnnotation(TraceAgent.class);
        if (agent != null) return agent.outputMapping();
        return null;
    }
}
