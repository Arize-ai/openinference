package com.arize.instrumentation.langchain4j;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.langchain4j.utils.SpanContext;
import dev.langchain4j.observability.api.event.AiServiceErrorEvent;
import dev.langchain4j.observability.api.listener.AiServiceErrorListener;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Scope;
import java.util.Map;

/**
 * Event listener for handling AI service error events in LangChain4j.
 * <p>
 * This listener records exceptions and marks the span as error when an AI service error event occurs.
 * </p>
 */
public class LangChain4jAiServiceErrorListener implements AiServiceErrorListener {

    private final OITracer tracer;
    private final Map<String, SpanContext> activeSpans;

    public LangChain4jAiServiceErrorListener(OITracer tracer, Map<String, SpanContext> activeSpans) {
        this.tracer = tracer;
        this.activeSpans = activeSpans;
    }

    /**
     * Handles the AI service error event by recording the exception and marking the span as errored.
     *
     * @param event the AI service error event
     */
    @Override
    public void onEvent(AiServiceErrorEvent event) {
        SpanContext spanContext =
                activeSpans.remove(event.invocationContext().invocationId().toString());
        if (spanContext != null) {
            Span span = spanContext.span();

            try (Scope scope = spanContext.context().makeCurrent()) {
                Throwable error = event.error();
                span.recordException(error);
                span.setStatus(StatusCode.ERROR, error.getMessage());
            } finally {
                span.end();
            }
        }
    }
}
