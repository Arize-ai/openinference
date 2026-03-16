package com.arize.instrumentation.langchain4j;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.langchain4j.utils.SpanContext;
import com.arize.semconv.trace.SemanticConventions;
import dev.langchain4j.observability.api.event.AiServiceCompletedEvent;
import dev.langchain4j.observability.api.listener.AiServiceCompletedListener;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;


import java.util.Map;
import java.util.UUID;

public class LangChain4jServiceCompletedListener implements AiServiceCompletedListener {

    private final OITracer tracer;
    private final Map<String, SpanContext> activeSpans;

    public LangChain4jServiceCompletedListener(OITracer tracer, Map<String, SpanContext> activeSpans) {
        this.tracer = tracer;
        this.activeSpans = activeSpans;
    }

    @Override
    public void onEvent(AiServiceCompletedEvent aiServiceCompletedEvent) {
        SpanContext spanContext = activeSpans.remove(aiServiceCompletedEvent.invocationContext().invocationId().toString());
        if (spanContext != null) {
            Span span = spanContext.span();
            span.setStatus(StatusCode.OK);
            span.setAttribute(SemanticConventions.OUTPUT_VALUE, "Execution Completed");
            span.end();
        }
    }
}
