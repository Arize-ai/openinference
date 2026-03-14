package com.arize.instrumentation.annotations;

import com.arize.instrumentation.OITracer;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Scope;
import java.util.List;
import java.util.Map;

public class TracedLLMSpan extends TracedSpan {

    private TracedLLMSpan(Span span, Scope scope, com.arize.instrumentation.TraceConfig config) {
        super(span, scope, config);
    }

    public static TracedLLMSpan start(OITracer tracer, String name) {
        Span span = startSpan(tracer, name, SemanticConventions.OpenInferenceSpanKind.LLM);
        Scope scope = span.makeCurrent();
        return new TracedLLMSpan(span, scope, tracer.getConfig());
    }

    public void setModelName(String model) {
        span.setAttribute(AttributeKey.stringKey(SemanticConventions.LLM_MODEL_NAME), model);
    }

    public void setSystem(SemanticConventions.LLMSystem system) {
        span.setAttribute(AttributeKey.stringKey(SemanticConventions.LLM_SYSTEM), system.getValue());
    }

    public void setProvider(SemanticConventions.LLMProvider provider) {
        span.setAttribute(AttributeKey.stringKey(SemanticConventions.LLM_PROVIDER), provider.getValue());
    }

    public void setInvocationParameters(Map<String, Object> params) {
        SpanHelper.SerializedValue sv = SpanHelper.serialize(params);
        if (sv != null) {
            span.setAttribute(AttributeKey.stringKey(SemanticConventions.LLM_INVOCATION_PARAMETERS), sv.value());
        }
    }

    public void setInputMessages(List<Map<String, Object>> messages) {
        if (config.isHideInputMessages()) return;
        SpanHelper.SerializedValue sv = SpanHelper.serialize(messages);
        if (sv != null) {
            span.setAttribute(AttributeKey.stringKey(SemanticConventions.LLM_INPUT_MESSAGES), sv.value());
        }
    }

    public void setOutputMessages(List<Map<String, Object>> messages) {
        if (config.isHideOutputMessages()) return;
        SpanHelper.SerializedValue sv = SpanHelper.serialize(messages);
        if (sv != null) {
            span.setAttribute(AttributeKey.stringKey(SemanticConventions.LLM_OUTPUT_MESSAGES), sv.value());
        }
    }

    public void setTokenCountPrompt(int count) {
        span.setAttribute(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_PROMPT), (long) count);
    }

    public void setTokenCountCompletion(int count) {
        span.setAttribute(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_COMPLETION), (long) count);
    }

    public void setTokenCountTotal(int count) {
        span.setAttribute(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_TOTAL), (long) count);
    }

    public void setCostPrompt(double cost) {
        span.setAttribute(AttributeKey.doubleKey(SemanticConventions.LLM_COST_PROMPT), cost);
    }

    public void setCostCompletion(double cost) {
        span.setAttribute(AttributeKey.doubleKey(SemanticConventions.LLM_COST_COMPLETION), cost);
    }

    public void setCostTotal(double cost) {
        span.setAttribute(AttributeKey.doubleKey(SemanticConventions.LLM_COST_TOTAL), cost);
    }
}
