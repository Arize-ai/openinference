package com.arize.instrumentation.langchain4j;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.langchain4j.utils.ChatMessageAttributeUtils;
import com.arize.instrumentation.langchain4j.utils.SpanContext;
import dev.langchain4j.model.chat.listener.ChatModelErrorContext;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.listener.ChatModelRequestContext;
import dev.langchain4j.model.chat.listener.ChatModelResponseContext;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.Scope;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 * Listener for LangChain4j chat models that creates OpenInference spans.
 */
public class LangChain4jModelListener implements ChatModelListener {

    private static final Logger logger = Logger.getLogger(LangChain4jModelListener.class.getName());

    private final OITracer tracer;
    private final Map<ChatRequest, SpanContext> activeSpans = new ConcurrentHashMap<>();

    public LangChain4jModelListener(OITracer tracer) {
        this.tracer = tracer;
    }

    @Override
    public void onRequest(ChatModelRequestContext requestContext) {
        ChatRequest request = requestContext.chatRequest();

        // Create span for the LLM call
        Span span = tracer.spanBuilder("generate")
                .setParent(Context.current())
                .setSpanKind(SpanKind.CLIENT)
                .startSpan();
        ChatMessageAttributeUtils.handleChatRequest(this.tracer, span, request);
        // Store span context for later use
        activeSpans.put(request, new SpanContext(span, Context.current().with(span)));
    }

    @Override
    public void onResponse(ChatModelResponseContext responseContext) {
        SpanContext spanContext = activeSpans.remove(responseContext.chatRequest());
        if (spanContext == null) {
            logger.warning("No active span found for response");
            return;
        }
        Span span = spanContext.span();
        ChatResponse response = responseContext.chatResponse();
        try (Scope scope = spanContext.context().makeCurrent()) {
            ChatMessageAttributeUtils.handleChatResponse(this.tracer, span, response);
        } finally {
            span.end();
        }
    }

    @Override
    public void onError(ChatModelErrorContext errorContext) {
        SpanContext spanContext = activeSpans.remove(errorContext.chatRequest());
        if (spanContext == null) {
            logger.warning("No active span found for error");
            return;
        }

        Span span = spanContext.span();

        try (Scope scope = spanContext.context().makeCurrent()) {
            Throwable error = errorContext.error();
            span.recordException(error);
            span.setStatus(StatusCode.ERROR, error.getMessage());
        } finally {
            span.end();
        }
    }

}
