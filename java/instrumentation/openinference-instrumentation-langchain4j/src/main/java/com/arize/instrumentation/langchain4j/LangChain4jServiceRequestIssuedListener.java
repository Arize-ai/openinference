package com.arize.instrumentation.langchain4j;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.langchain4j.utils.ChatMessageAttributeUtils;
import com.arize.instrumentation.langchain4j.utils.SpanContext;
import com.arize.semconv.trace.SemanticConventions;
import dev.langchain4j.observability.api.event.AiServiceRequestIssuedEvent;
import dev.langchain4j.observability.api.event.AiServiceResponseReceivedEvent;
import dev.langchain4j.observability.api.listener.AiServiceRequestIssuedListener;
import dev.langchain4j.observability.api.listener.AiServiceResponseReceivedListener;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Context;

import java.util.Map;
import java.util.UUID;

public class LangChain4jServiceRequestIssuedListener implements AiServiceRequestIssuedListener {

    private final OITracer tracer;
    private final Map<String, SpanContext> activeSpans;

    public LangChain4jServiceRequestIssuedListener(OITracer tracer, Map<String, SpanContext> activeSpans) {
        this.tracer = tracer;
        this.activeSpans = activeSpans;
    }

    @Override
    public void onEvent(AiServiceRequestIssuedEvent event) {
        UUID invocationId = event.invocationContext().invocationId();
        SpanContext parentSpanContext = activeSpans.get(invocationId.toString());
        Context context = parentSpanContext != null
                ? parentSpanContext.context() // Use AiService span's context
                : Context.current();
        Span span = tracer.spanBuilder("AiService.chat")
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
