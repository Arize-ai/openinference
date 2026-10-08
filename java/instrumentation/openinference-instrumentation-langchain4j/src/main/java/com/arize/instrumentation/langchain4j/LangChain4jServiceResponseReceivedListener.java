package com.arize.instrumentation.langchain4j;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.langchain4j.utils.ChatMessageAttributeUtils;
import com.arize.instrumentation.langchain4j.utils.SpanContext;
import dev.langchain4j.observability.api.event.AiServiceResponseReceivedEvent;
import dev.langchain4j.observability.api.listener.AiServiceResponseReceivedListener;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Scope;
import java.util.Map;

/**
 * Event listener for handling AI service response received events in LangChain4j.
 * <p>
 * This listener ends the tracing span for a chat response and records response attributes.
 * </p>
 */
public class LangChain4jServiceResponseReceivedListener implements AiServiceResponseReceivedListener {

    private final OITracer tracer;
    private final Map<String, SpanContext> activeSpans;
    private final Map<String, SpanContext> llmSpans;

    public LangChain4jServiceResponseReceivedListener(
            OITracer tracer, Map<String, SpanContext> activeSpans, Map<String, SpanContext> llmSpans) {
        this.tracer = tracer;
        this.activeSpans = activeSpans;
        this.llmSpans = llmSpans;
    }

    /**
     * Handles the AI service response received event by ending the span and recording response attributes.
     *
     * @param event the AI service response received event
     */
    @Override
    public void onEvent(AiServiceResponseReceivedEvent event) {
        String invocationId = event.invocationContext().invocationId().toString();
        SpanContext spanContext = llmSpans.remove(invocationId);
        if (spanContext != null) {
            Span span = spanContext.span();
            try (Scope scope = spanContext.context().makeCurrent()) {
                ChatMessageAttributeUtils.handleChatResponse(this.tracer, span, event.response());
            } finally {
                span.end();
            }
        }
    }
}
