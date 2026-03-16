package com.arize.instrumentation.langchain4j;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.langchain4j.utils.SpanContext;
import com.arize.semconv.trace.SemanticConventions;
import dev.langchain4j.observability.api.event.AiServiceCompletedEvent;
import dev.langchain4j.observability.api.event.AiServiceResponseReceivedEvent;
import dev.langchain4j.observability.api.listener.AiServiceCompletedListener;
import dev.langchain4j.observability.api.listener.AiServiceResponseReceivedListener;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Context;

import java.util.Map;
import java.util.UUID;

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
            span.setAttribute(SemanticConventions.OUTPUT_VALUE, event.response().toString());
            span.setStatus(StatusCode.OK);
            span.end();
        }
    }
}
