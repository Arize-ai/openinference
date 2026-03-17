package com.arize.instrumentation.langchain4j;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.langchain4j.utils.SpanContext;
import dev.langchain4j.observability.api.event.AiServiceErrorEvent;
import dev.langchain4j.observability.api.listener.AiServiceErrorListener;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Scope;

import java.util.Map;

public class LangChain4jAiServiceErrorListener implements AiServiceErrorListener {

    private final OITracer tracer;
    private final Map<String, SpanContext> activeSpans;

    public LangChain4jAiServiceErrorListener(OITracer tracer, Map<String, SpanContext> activeSpans) {
        this.tracer = tracer;
        this.activeSpans = activeSpans;
    }

    @Override
    public void onEvent(AiServiceErrorEvent event) {
        SpanContext spanContext = activeSpans.remove(event.invocationContext().invocationId().toString());
        if (spanContext != null) {
            Span span = spanContext.span();

            try (Scope scope = spanContext.context().makeCurrent()) {
                Throwable error = event.error();
                span.recordException(error);
                span.setStatus(StatusCode.ERROR);
            } finally {
                span.end();
            }
        }
    }
}
