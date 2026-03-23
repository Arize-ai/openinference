package com.arize.instrumentation.trace;

import com.arize.instrumentation.OITracer;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Scope;
import java.util.List;
import java.util.Map;

public class LLMSpan extends TracedSpan {

    private LLMSpan(Span span, Scope scope, com.arize.instrumentation.TraceConfig config) {
        super(span, scope, config);
    }

    public static LLMSpan start(OITracer tracer, String name) {
        Span span = startSpan(tracer, name, SemanticConventions.OpenInferenceSpanKind.LLM);
        Scope scope = span.makeCurrent();
        return new LLMSpan(span, scope, tracer.getConfig());
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
        SpanSerializer.SerializedValue sv = SpanSerializer.serialize(params);
        if (sv != null) {
            span.setAttribute(AttributeKey.stringKey(SemanticConventions.LLM_INVOCATION_PARAMETERS), sv.value());
        }
    }

    public void setInputMessages(List<Map<String, Object>> messages) {
        if (config.isHideInputMessages()) return;
        SpanSerializer.SerializedValue sv = SpanSerializer.serialize(messages);
        if (sv != null) {
            span.setAttribute(AttributeKey.stringKey(SemanticConventions.LLM_INPUT_MESSAGES), sv.value());
        }
    }

    public void setOutputMessages(List<Map<String, Object>> messages) {
        if (config.isHideOutputMessages()) return;
        SpanSerializer.SerializedValue sv = SpanSerializer.serialize(messages);
        if (sv != null) {
            span.setAttribute(AttributeKey.stringKey(SemanticConventions.LLM_OUTPUT_MESSAGES), sv.value());
        }
    }

    public void setTokenCountPrompt(long count) {
        span.setAttribute(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_PROMPT), count);
    }

    public void setTokenCountCompletion(long count) {
        span.setAttribute(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_COMPLETION), count);
    }

    public void setTokenCountTotal(long count) {
        span.setAttribute(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_TOTAL), count);
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
