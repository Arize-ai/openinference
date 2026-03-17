package com.arize.instrumentation.langchain4j;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.langchain4j.utils.ChatMessageAttributeUtils;
import com.arize.instrumentation.langchain4j.utils.SpanContext;
import com.arize.semconv.trace.SemanticConventions;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.observability.api.event.AiServiceStartedEvent;
import dev.langchain4j.observability.api.listener.AiServiceStartedListener;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.context.Context;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class LangChain4jAiServiceStartedListener implements AiServiceStartedListener {

    private final OITracer tracer;
    private final Map<String, SpanContext> activeSpans;

    public LangChain4jAiServiceStartedListener(OITracer tracer, Map<String, SpanContext> activeSpans) {
        this.tracer = tracer;
        this.activeSpans = activeSpans;
    }

    public void setInputAttributes(Span span, AiServiceStartedEvent event) {
        span.setAttribute(
                SemanticConventions.OPENINFERENCE_SPAN_KIND,
                SemanticConventions.OpenInferenceSpanKind.CHAIN.getValue()
        );
        List<ChatMessage> messages = new ArrayList<>();
        if (event.systemMessage().isPresent()) {
            messages.add(event.systemMessage().get());
        }
        messages.add(event.userMessage());
        List<Map<String, Object>> inputList = ChatMessageAttributeUtils.convertMessages(messages);
        ObjectMapper objectMapper = new ObjectMapper();
        try {
            span.setAttribute(SemanticConventions.INPUT_VALUE, objectMapper.writeValueAsString(inputList));
        } catch (Exception e) {
            span.setAttribute(SemanticConventions.INPUT_VALUE, event.userMessage().toString());
        }
        span.setAttribute(
                SemanticConventions.INPUT_MIME_TYPE,
                SemanticConventions.MimeType.JSON.getValue()
        );
    }

    @Override
    public void onEvent(AiServiceStartedEvent event) {
        Span span = tracer.spanBuilder("AiService.chat")
                .setParent(Context.current())
                .setSpanKind(SpanKind.CLIENT)
                .startSpan();

        // Set basic attributes
        setInputAttributes(span, event);
        activeSpans.put(
                event.invocationContext().invocationId().toString(),
                new SpanContext(span, Context.current().with(span))
        );
    }
}
