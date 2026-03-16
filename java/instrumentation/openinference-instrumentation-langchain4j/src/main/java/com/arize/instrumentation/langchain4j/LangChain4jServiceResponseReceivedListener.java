package com.arize.instrumentation.langchain4j;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.langchain4j.utils.ChatMessageAttributeUtils;
import com.arize.instrumentation.langchain4j.utils.SpanContext;
import dev.langchain4j.observability.api.event.AiServiceResponseReceivedEvent;
import dev.langchain4j.observability.api.listener.AiServiceResponseReceivedListener;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Scope;

import java.util.Map;

public class LangChain4jServiceResponseReceivedListener implements AiServiceResponseReceivedListener {

    private final OITracer tracer;
    private final Map<String, SpanContext> activeSpans;

    public LangChain4jServiceResponseReceivedListener(OITracer tracer, Map<String, SpanContext> activeSpans) {
        this.tracer = tracer;
        this.activeSpans = activeSpans;
    }

    @Override
    public void onEvent(AiServiceResponseReceivedEvent event) {
        String invocationId = event.invocationContext().invocationId().toString();
        SpanContext spanContext = activeSpans.remove("chat_" + invocationId);
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
