package com.arize.instrumentation.langchain4j;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.langchain4j.utils.ChatMessageAttributeUtils;
import com.arize.instrumentation.langchain4j.utils.SpanContext;
import dev.langchain4j.observability.api.event.AiServiceRequestIssuedEvent;
import dev.langchain4j.observability.api.listener.AiServiceRequestIssuedListener;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.context.Context;

import java.util.Map;
import java.util.UUID;

/**
 * Event listener for handling AI service request issued events in LangChain4j.
 * <p>
 * This listener creates a tracing span for each chat request and records request attributes.
 * </p>
 */
public class LangChain4jServiceRequestIssuedListener implements AiServiceRequestIssuedListener {

    private final OITracer tracer;
    private final Map<String, SpanContext> activeSpans;

    public LangChain4jServiceRequestIssuedListener(OITracer tracer, Map<String, SpanContext> activeSpans) {
        this.tracer = tracer;
        this.activeSpans = activeSpans;
    }

    /**
     * Handles the AI service request issued event by creating a span and recording request attributes.
     *
     * @param event the AI service request issued event
     */
    @Override
    public void onEvent(AiServiceRequestIssuedEvent event) {
        UUID invocationId = event.invocationContext().invocationId();
        SpanContext parentSpanContext = activeSpans.get(invocationId.toString());
        Context context = parentSpanContext != null
                ? parentSpanContext.context() // Use AiService span's context
                : Context.current();
        Span span = tracer.spanBuilder("LLM")
                .setParent(context)
                .setSpanKind(SpanKind.CLIENT)
                .startSpan();
        ChatMessageAttributeUtils.handleChatRequest(this.tracer, span, event.request());
        activeSpans.put(
                "chat_" + event.invocationContext().invocationId().toString(),
                new SpanContext(span, Context.current().with(span))
        );
    }
}
